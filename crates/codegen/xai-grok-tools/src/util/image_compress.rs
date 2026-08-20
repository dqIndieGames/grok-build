//! Shared image encoding for the coding-model send path and Imagine.
//!
//! Prompt images (paste / `read_file` / `image_edit` references) use Codex
//! `high` limits via [`resize_and_encode_prompt_image`].
//! [`re_encode_under_limit`] remains a byte-capped encoder for other callers.

use std::borrow::Cow;

use image::DynamicImage;
use image::ImageFormat;
use image::codecs::jpeg::JpegEncoder;
pub use image::imageops::FilterType;

/// Parameters that control the re-encode loop.
///
/// Each call-site provides its own set of limits so the shared encoder can
/// serve callers with different size/quality trade-offs.
#[derive(Debug, Clone, Copy)]
pub struct ReEncodeParams {
    /// Maximum output size in **raw bytes** (not base64).
    pub max_bytes: usize,

    /// Maximum dimension (width or height) on the first attempt.
    pub max_side_px: u32,

    /// Maximum total output pixel count (width × height) on the first
    /// attempt; `u64::MAX` disables the area cap.
    pub max_pixels: u64,

    /// Floor dimension — the loop gives up when `max_side` falls to or below
    /// this value without producing output that fits.
    pub min_side_px: u32,

    /// JPEG quality steps to try at each dimension, in descending order.
    pub quality_steps: &'static [u8],

    /// Resize filter (e.g. `CatmullRom`, `Lanczos3`).
    pub filter: FilterType,
}

impl ReEncodeParams {
    /// True when either side exceeds `max_side_px` or the total pixel count
    /// exceeds `max_pixels` — shared by re-encode triggers and passthrough
    /// gates so the rule cannot drift between them.
    pub fn exceeds_dimension_caps(&self, w: u32, h: u32) -> bool {
        w > self.max_side_px
            || h > self.max_side_px
            || u64::from(w) * u64::from(h) > self.max_pixels
    }
}

/// Vision-token patch size used by Codex / OpenAI Responses image math.
pub const PROMPT_IMAGE_PATCH_SIZE: u32 = 32;
/// Codex `detail: high` longest-side cap.
pub const HIGH_MAX_DIMENSION: u32 = 2048;
/// Codex `detail: high` patch-count cap (`ceil(w/32) * ceil(h/32)`).
pub const HIGH_MAX_PATCHES: usize = 2_500;
/// Sanity guard against pathological inputs, matching Codex
/// `MAX_PROMPT_IMAGE_INPUT_BYTES`. Not a target upload size.
pub const MAX_PROMPT_IMAGE_INPUT_BYTES: usize = 1024 * 1024 * 1024;
const PROMPT_JPEG_QUALITY: u8 = 85;
const PROMPT_RESIZE_FILTER: FilterType = FilterType::Triangle;

/// Dimension + patch budget for images sent to the coding model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PromptImageResizeLimits {
    pub max_dimension: u32,
    pub max_patches: usize,
}

/// Codex `HIGH_DETAIL_LIMITS` (image_preparation.rs).
pub const HIGH_DETAIL_LIMITS: PromptImageResizeLimits = PromptImageResizeLimits {
    max_dimension: HIGH_MAX_DIMENSION,
    max_patches: HIGH_MAX_PATCHES,
};

/// Cursor-harness side-only clamp: 1024px, no patch budget.
pub const STRICT_DETAIL_LIMITS: PromptImageResizeLimits = PromptImageResizeLimits {
    max_dimension: 1024,
    max_patches: usize::MAX,
};

/// True when `width`×`height` already fit `limits` (Codex `prompt_image_dimensions_fit`).
pub fn prompt_image_dimensions_fit(
    width: u32,
    height: u32,
    limits: PromptImageResizeLimits,
) -> bool {
    let patches_wide = width.div_ceil(PROMPT_IMAGE_PATCH_SIZE);
    let patches_high = height.div_ceil(PROMPT_IMAGE_PATCH_SIZE);
    let patch_count = u64::from(patches_wide) * u64::from(patches_high);
    width <= limits.max_dimension
        && height <= limits.max_dimension
        && patch_count <= limits.max_patches as u64
}

/// Target width/height under Codex `prompt_image_output_dimensions_for_limits`.
/// Never upscales.
pub fn prompt_image_output_dimensions(
    width: u32,
    height: u32,
    limits: PromptImageResizeLimits,
) -> (u32, u32) {
    let width = width.max(1);
    let height = height.max(1);
    if prompt_image_dimensions_fit(width, height, limits) {
        return (width, height);
    }

    let max_dimension_scale =
        (f64::from(limits.max_dimension) / f64::from(width.max(height))).min(1.0);
    let width = ((f64::from(width) * max_dimension_scale).round() as u32).max(1);
    let height = ((f64::from(height) * max_dimension_scale).round() as u32).max(1);
    if prompt_image_dimensions_fit(width, height, limits) {
        return (width, height);
    }

    let width_f64 = f64::from(width);
    let height_f64 = f64::from(height);
    let patch_size = f64::from(PROMPT_IMAGE_PATCH_SIZE);
    let mut scale =
        (patch_size * patch_size * limits.max_patches as f64 / width_f64 / height_f64).sqrt();
    let scaled_patches_wide = width_f64 * scale / patch_size;
    let scaled_patches_high = height_f64 * scale / patch_size;
    scale *= (scaled_patches_wide.floor() / scaled_patches_wide)
        .min(scaled_patches_high.floor() / scaled_patches_high);

    (
        ((width_f64 * scale).floor() as u32).max(1),
        ((height_f64 * scale).floor() as u32).max(1),
    )
}

/// Prefer keeping JPEG/WebP when those are the source; everything else becomes PNG.
pub fn preferred_prompt_format(bytes: &[u8]) -> ImageFormat {
    match image::guess_format(bytes) {
        Ok(ImageFormat::Jpeg) => ImageFormat::Jpeg,
        Ok(ImageFormat::WebP) => ImageFormat::WebP,
        _ => ImageFormat::Png,
    }
}

fn encode_prompt_image(
    image: &DynamicImage,
    preferred: ImageFormat,
) -> Result<(Vec<u8>, &'static str), ReEncodeError> {
    let target = match preferred {
        ImageFormat::Jpeg => ImageFormat::Jpeg,
        ImageFormat::WebP => ImageFormat::WebP,
        _ => ImageFormat::Png,
    };
    let mut buffer = Vec::new();
    match target {
        ImageFormat::Jpeg => {
            let mut enc = JpegEncoder::new_with_quality(&mut buffer, PROMPT_JPEG_QUALITY);
            enc.encode_image(image).map_err(|_| ReEncodeError::EncodeFailed)?;
        }
        ImageFormat::WebP => {
            image
                .write_to(&mut std::io::Cursor::new(&mut buffer), ImageFormat::WebP)
                .map_err(|_| ReEncodeError::EncodeFailed)?;
        }
        _ => {
            image
                .write_to(&mut std::io::Cursor::new(&mut buffer), ImageFormat::Png)
                .map_err(|_| ReEncodeError::EncodeFailed)?;
        }
    }
    let mime = match target {
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::WebP => "image/webp",
        _ => "image/png",
    };
    Ok((buffer, mime))
}

/// Resize to the Codex budget (if needed) and re-encode. No per-image byte target:
/// quality is fixed at JPEG 85 / PNG / WebP. Callers that already fit should
/// pass through original bytes instead of calling this.
pub fn resize_and_encode_prompt_image(
    decoded: &DynamicImage,
    preferred: ImageFormat,
    limits: PromptImageResizeLimits,
) -> Result<(Vec<u8>, u32, u32, &'static str), ReEncodeError> {
    let (target_w, target_h) =
        prompt_image_output_dimensions(decoded.width(), decoded.height(), limits);
    let scaled: Cow<'_, DynamicImage> =
        if target_w == decoded.width() && target_h == decoded.height() {
            Cow::Borrowed(decoded)
        } else {
            Cow::Owned(decoded.resize_exact(target_w, target_h, PROMPT_RESIZE_FILTER))
        };
    let (bytes, mime) = encode_prompt_image(&scaled, preferred)?;
    Ok((bytes, target_w, target_h, mime))
}

/// Why `re_encode_under_limit` could not produce a compliant output.
#[derive(Debug, thiserror::Error)]
pub enum ReEncodeError {
    /// Exhausted all quality × dimension steps without fitting under the cap.
    #[error(
        "re-encode could not fit under {max_bytes} bytes after PNG+JPEG attempts (last side {last_side}px)"
    )]
    CouldNotFit { max_bytes: usize, last_side: u32 },
    /// Prompt-image PNG/JPEG/WebP encode failed.
    #[error("prompt image encode failed")]
    EncodeFailed,
}

/// Try PNG and JPEG encodings at descending dimensions, returning whichever
/// is smallest and fits under `params.max_bytes`.
///
/// On success returns `(bytes, width, height, mime_type)`.
pub fn re_encode_under_limit(
    decoded: &DynamicImage,
    params: &ReEncodeParams,
) -> Result<(Vec<u8>, u32, u32, &'static str), ReEncodeError> {
    // Never upscale: a small-but-heavy image is re-encoded at its own
    // resolution, not enlarged to `max_side_px`. `image::resize` scales *up* to
    // fill the target box, so starting at `max_side_px` would enlarge anything
    // smaller — adding no detail and wasting request bytes / cache headroom.
    let original_max_side = decoded.width().max(decoded.height());
    let mut max_side = params.max_side_px.min(original_max_side);
    let original_pixels = u64::from(decoded.width()) * u64::from(decoded.height());
    if original_pixels > params.max_pixels {
        max_side = max_side.min(area_capped_side(
            original_max_side,
            decoded.width().min(decoded.height()),
            params.max_pixels,
        ));
    }

    loop {
        // Only resample when actually downscaling; resizing to the current size
        // would just soften the image for no reason. `resize(w, h)` preserves
        // aspect ratio (fits inside w×h, not stretch-to-square).
        let scaled: Cow<'_, DynamicImage> = if max_side < original_max_side {
            Cow::Owned(decoded.resize(max_side, max_side, params.filter))
        } else {
            Cow::Borrowed(decoded)
        };
        let img: &DynamicImage = &scaled;
        let (w, h) = (img.width(), img.height());

        // --- PNG candidate ---------------------------------------------------
        let png_candidate = {
            let mut buf = Vec::new();
            img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
                .ok()
                .filter(|_| buf.len() <= params.max_bytes)
                .map(|_| buf)
        };

        // --- JPEG candidate (best quality that fits) -------------------------
        let jpeg_candidate = params.quality_steps.iter().find_map(|&quality| {
            let mut buf = Vec::new();
            let mut enc = JpegEncoder::new_with_quality(&mut buf, quality);
            enc.encode_image(img).ok()?;
            (buf.len() <= params.max_bytes).then_some(buf)
        });

        // --- Pick the smaller candidate --------------------------------------
        match (png_candidate, jpeg_candidate) {
            (Some(png), Some(jpeg)) => {
                if png.len() <= jpeg.len() {
                    return Ok((png, w, h, "image/png"));
                } else {
                    return Ok((jpeg, w, h, "image/jpeg"));
                }
            }
            (Some(png), None) => return Ok((png, w, h, "image/png")),
            (None, Some(jpeg)) => return Ok((jpeg, w, h, "image/jpeg")),
            (None, None) => { /* fall through to smaller dimensions */ }
        }

        if max_side <= params.min_side_px {
            return Err(ReEncodeError::CouldNotFit {
                max_bytes: params.max_bytes,
                last_side: max_side,
            });
        }
        max_side = max_side * 3 / 4;
    }
}

/// Largest target long side whose resize output area stays within `max_pixels`.
fn area_capped_side(long: u32, short: u32, max_pixels: u64) -> u32 {
    let scale = (max_pixels as f64 / (u64::from(long) * u64::from(short)) as f64).sqrt();
    let mut side = ((f64::from(long) * scale).floor() as u32).clamp(1, long);
    // Nearest-rounding of the short side can overshoot the budget by ~side/2
    // pixels, so step down until the predicted output fits: a decrement removes
    // ~2*area/side pixels, giving ~2 iterations for ordinary aspect ratios;
    // only degenerate strips whose short side pins at the 1px floor walk
    // O(side), bounded by the callers' decode-pixel limits.
    while side > 1 && predicted_resize_area(long, short, side) > max_pixels {
        side -= 1;
    }
    side
}

/// Output area `image::resize` produces for a `side`×`side` bounding box,
/// mirroring `resize_dimensions` (image-0.25.9, `src/math/utils.rs`)
/// expression-for-expression; the `area_cap_exact_fit_across_aspect_ratios`
/// sweep pins the equivalence through the real resize, so a crate bump that
/// changes the rounding shows up as a test failure pointing here.
fn predicted_resize_area(long: u32, short: u32, side: u32) -> u64 {
    let ratio = f64::from(side) / f64::from(long);
    let scaled_long = (f64::from(long) * ratio).round().max(1.0) as u64;
    let scaled_short = (f64::from(short) * ratio).round().max(1.0) as u64;
    scaled_long * scaled_short
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};

    use super::*;

    /// Deterministic high-entropy image so PNG/JPEG can't trivially shrink it.
    fn noise(w: u32, h: u32) -> DynamicImage {
        let mut img = RgbImage::new(w, h);
        let mut s: u32 = 0x1234_5678;
        for p in img.pixels_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *p = Rgb([(s >> 16) as u8, (s >> 8) as u8, s as u8]);
        }
        DynamicImage::ImageRgb8(img)
    }

    fn params(max_bytes: usize, max_side_px: u32, max_pixels: u64) -> ReEncodeParams {
        ReEncodeParams {
            max_bytes,
            max_side_px,
            max_pixels,
            min_side_px: 256,
            quality_steps: &[88, 72, 56, 40, 24],
            filter: FilterType::CatmullRom,
        }
    }

    #[test]
    fn does_not_upscale_images_smaller_than_the_side_cap() {
        // 1280x960 is already under the test's 1568px side cap. Re-encoding
        // must NOT enlarge it — output dimensions must never exceed the input.
        // (Regression: the resize previously scaled small images up to
        // `max_side_px`.)
        let img = noise(1280, 960);
        let (_bytes, w, h, _mime) =
            re_encode_under_limit(&img, &params(5_000_000, 1568, u64::MAX)).unwrap();
        assert!(
            w <= 1280 && h <= 960,
            "must not upscale a 1280x960 image, got {w}x{h}"
        );
    }

    #[test]
    fn downscales_images_larger_than_the_side_cap() {
        // 2000x1500 exceeds the test's 1568px side cap and must fit inside it.
        let img = noise(2000, 1500);
        let (_bytes, w, h, _mime) =
            re_encode_under_limit(&img, &params(5_000_000, 1568, u64::MAX)).unwrap();
        assert!(
            w <= 1568 && h <= 1568,
            "should downscale to <=1568, got {w}x{h}"
        );
        assert_eq!(w, 1568, "longest side should hit the cap");
    }

    #[test]
    fn shrinks_dimensions_only_when_bytes_force_it() {
        // A small image that can't fit the byte cap at native size is
        // downscaled below its own dimensions — still never above them.
        let img = noise(1280, 960);
        let (bytes, w, h, _mime) =
            re_encode_under_limit(&img, &params(120_000, 1568, u64::MAX)).unwrap();
        assert!(bytes.len() <= 120_000);
        assert!(w <= 1280 && h <= 960, "must not upscale, got {w}x{h}");
    }

    #[test]
    fn area_cap_bounds_total_pixels_for_wide_images() {
        // 3438x1830 = 6.29 Mpx: the side cap is loose, so only the area budget
        // binds; expected long side = floor(3438 * sqrt(2_408_448 / 6_291_540)).
        let img = noise(3438, 1830);
        let (_bytes, w, h, _mime) =
            re_encode_under_limit(&img, &params(50_000_000, 10_000, 2_408_448)).unwrap();
        let area = u64::from(w) * u64::from(h);
        assert!(area <= 2_408_448, "area {area} over budget ({w}x{h})");
        assert!(
            area >= 2_300_000,
            "should use most of the budget, got {area} ({w}x{h})"
        );
        assert_eq!(w, 2127, "long side ~2127 for a 3438x1830 source");
        let r_in = 3438.0 / 1830.0;
        let r_out = w as f64 / h as f64;
        assert!(
            (r_in - r_out).abs() < 0.01,
            "aspect ratio {r_in} -> {r_out} ({w}x{h})"
        );
    }

    #[test]
    fn image_under_area_cap_is_not_resized() {
        // 1500x1500 = 2.25 Mpx is under the 2_408_448 budget; no resample.
        let img = noise(1500, 1500);
        let (_bytes, w, h, _mime) =
            re_encode_under_limit(&img, &params(50_000_000, 10_000, 2_408_448)).unwrap();
        assert_eq!((w, h), (1500, 1500), "must not up- or downscale");
    }

    #[test]
    fn area_cap_exact_fit_across_aspect_ratios() {
        // Short-side rounding must never push the output area over the cap;
        // (1600, 400, 300_000) rounds 273.5 up and exercises the decrement.
        for &(sw, sh, cap) in &[
            (1300u32, 900u32, 500_000u64),
            (1200, 1199, 640_000),
            (1600, 400, 300_000),
            (900, 1300, 777_777),
            (1000, 1000, 123_456),
            (2600, 1800, 2_408_448),
        ] {
            let img = noise(sw, sh);
            let (_bytes, w, h, _mime) =
                re_encode_under_limit(&img, &params(50_000_000, 10_000, cap)).unwrap();
            let area = u64::from(w) * u64::from(h);
            assert!(area <= cap, "{sw}x{sh} cap {cap}: got {w}x{h} = {area}");
            assert_eq!(
                w.max(h),
                area_capped_side(sw.max(sh), sw.min(sh), cap),
                "{sw}x{sh} cap {cap}: long side must match the predicted fit"
            );
        }
    }

    // Ground truth: Codex
    // `codex-rs/utils/image/src/image_tests.rs`
    // `resize_with_limits_respects_dimension_and_patch_budgets`
    // and `codex-rs/core/src/image_preparation_tests.rs`
    // `detail_policies_apply_the_expected_budgets` (High, 2048x2048 → 1600x1600).
    #[test]
    fn high_detail_2048_square_matches_codex() {
        assert_eq!(
            prompt_image_output_dimensions(2048, 2048, HIGH_DETAIL_LIMITS),
            (1600, 1600)
        );
        let img = noise(2048, 2048);
        let (_bytes, w, h, _mime) =
            resize_and_encode_prompt_image(&img, ImageFormat::Png, HIGH_DETAIL_LIMITS).unwrap();
        assert_eq!((w, h), (1600, 1600));
        assert!(prompt_image_dimensions_fit(w, h, HIGH_DETAIL_LIMITS));
    }

    #[test]
    fn high_detail_output_fits_and_never_upscales() {
        for &(sw, sh) in &[
            (1500u32, 1500u32),
            (1600, 600),
            (3438, 1830),
            (3000, 2000),
            (1700, 1700),
            (1800, 1700),
            (3840, 2160),
            (4096, 2048),
            (1024, 4096),
        ] {
            let (w, h) = prompt_image_output_dimensions(sw, sh, HIGH_DETAIL_LIMITS);
            assert!(
                prompt_image_dimensions_fit(w, h, HIGH_DETAIL_LIMITS),
                "{sw}x{sh} -> {w}x{h} must fit high"
            );
            assert!(w <= sw && h <= sh, "{sw}x{sh} -> {w}x{h} must not upscale");
        }
        assert!(prompt_image_dimensions_fit(1500, 1500, HIGH_DETAIL_LIMITS));
        assert!(prompt_image_dimensions_fit(1600, 600, HIGH_DETAIL_LIMITS));
        assert!(!prompt_image_dimensions_fit(2048, 2048, HIGH_DETAIL_LIMITS));
        assert!(!prompt_image_dimensions_fit(1700, 1700, HIGH_DETAIL_LIMITS));
    }

    #[test]
    fn high_detail_preserves_jpeg_mime_when_resizing() {
        let img = noise(3000, 2000);
        let (_bytes, w, h, mime) =
            resize_and_encode_prompt_image(&img, ImageFormat::Jpeg, HIGH_DETAIL_LIMITS).unwrap();
        assert_eq!(mime, "image/jpeg");
        assert!(prompt_image_dimensions_fit(w, h, HIGH_DETAIL_LIMITS));
        assert!(w.max(h) <= HIGH_MAX_DIMENSION);
    }
}
