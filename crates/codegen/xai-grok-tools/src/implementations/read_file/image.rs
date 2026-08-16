//! Image compression for conversation embedding.

/// Why [`compress_image_for_conversation`] could not produce a
/// model-embeddable image.
#[derive(Debug, thiserror::Error)]
pub enum CompressImageError {
    /// Pre-decode pixel-area cap exceeded.
    #[error("image dimensions {width}x{height} exceed the {limit_pixels} pixel decode limit")]
    PixelLimitExceeded {
        width: u32,
        height: u32,
        limit_pixels: u64,
    },
    /// Lowest quality / smallest dimension still exceeded the byte cap.
    /// Unused on the Codex-high prompt path (no per-image byte target); kept
    /// so Imagine / callers matching this variant still compile.
    #[error("compressed image still exceeds the {0}-byte conversation payload cap")]
    PayloadCapExceeded(usize),
    /// Decoded input exceeded Codex `MAX_PROMPT_IMAGE_INPUT_BYTES`.
    #[error("image input is too large ({size} bytes; max {limit} bytes)")]
    InputTooLarge { size: usize, limit: usize },
    /// No recognised magic bytes, or the IO reader failed before header read.
    #[error("image format could not be detected")]
    FormatDetectionFailed,
    /// Format detected but pixel decode failed (CRC, IDAT truncation, ...).
    #[error("image decode failed: {0}")]
    DecodeFailed(String),
}

/// Legacy fixture size for garbage-input tests. Not a send-time byte gate.
pub const MAX_IMAGE_PAYLOAD_BYTES: usize = 768 * 1024;

/// Absolute upper bound on decoded pixel count before we refuse to decode.
/// Matches the model API's `MAX_IMAGE_PIXELS` ceiling (and the shell's
/// `MAX_VISION_TOTAL_PX`) so any photo the API would accept can be read and
/// downscaled — a 20-48 Mpx camera photo must not fail `read_file`. Images
/// above this are rejected by the API regardless.
const MAX_DECODE_PIXELS: u64 = 178_956_970;

/// Prepare a disk image for the coding model using Codex `high` limits
/// (2048 long side, 2500 vision patches). Dimensions that already fit are
/// passed through as original JPEG/PNG/WebP bytes.
pub fn compress_image_for_conversation(
    raw_bytes: Vec<u8>,
    original_mime: String,
) -> Result<(Vec<u8>, String), CompressImageError> {
    compress_image_for_conversation_high(raw_bytes, original_mime)
}

/// [`compress_image_for_conversation`] off the async path, mapped to the
/// read tools' output: an embeddable
/// [`ImageContent`](crate::types::output::ImageContent) on success, or
/// [`ImageSizeError`](crate::types::output::ReadFileOutput::ImageSizeError)
/// with the model-visible reason.
pub async fn image_read_output(
    file_bytes: Vec<u8>,
    mime_type: String,
) -> crate::types::output::ReadFileOutput {
    use crate::types::output::{ImageContent, ReadFileOutput};
    use base64::Engine as _;

    let (encoded_bytes, mime) = match tokio::task::spawn_blocking(move || {
        compress_image_for_conversation(file_bytes, mime_type)
    })
    .await
    {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => {
            return ReadFileOutput::ImageSizeError(format!(
                "Could not embed image in conversation: {e}"
            ));
        }
        Err(e) => {
            // Don't leak `JoinError::Display` (panic payload / paths)
            // into model-visible text.
            tracing::warn!(error = %e, "image compression task panicked");
            return ReadFileOutput::ImageSizeError(
                "Image compression failed; see logs.".to_owned(),
            );
        }
    };
    ReadFileOutput::ImageContent(ImageContent {
        data: base64::engine::general_purpose::STANDARD.encode(&encoded_bytes),
        mime_type: mime,
        annotations: None,
        uri: None,
        meta: None,
    })
}

/// Codex-high body: GIF/BMP/TIFF → PNG, then passthrough or Triangle resize.
fn compress_image_for_conversation_high(
    raw_bytes: Vec<u8>,
    original_mime: String,
) -> Result<(Vec<u8>, String), CompressImageError> {
    use crate::util::image_compress::{
        HIGH_DETAIL_LIMITS, MAX_PROMPT_IMAGE_INPUT_BYTES, ReEncodeError,
        preferred_prompt_format, prompt_image_dimensions_fit, resize_and_encode_prompt_image,
    };
    use image::ImageReader;
    use std::io::Cursor;

    if raw_bytes.len() > MAX_PROMPT_IMAGE_INPUT_BYTES {
        return Err(CompressImageError::InputTooLarge {
            size: raw_bytes.len(),
            limit: MAX_PROMPT_IMAGE_INPUT_BYTES,
        });
    }

    // Engines only sample JPEG/PNG/WebP; PNG ICO/GIF/BMP/TIFF here. Before the
    // small-image early return so we keep the converted bytes.
    let (raw_bytes, original_mime) =
        match crate::util::image_validate::transcode_to_endpoint_png(&raw_bytes) {
            Some(Ok(png)) => (png, "image/png".to_string()),
            Some(Err(e)) => {
                tracing::warn!(
                    error = %e,
                    "non-native image format transcode to PNG failed; cannot embed image"
                );
                return Err(CompressImageError::DecodeFailed(format!(
                    "non-native image format transcode failed: {e}"
                )));
            }
            None => (raw_bytes, original_mime),
        };

    if raw_bytes.len() > MAX_PROMPT_IMAGE_INPUT_BYTES {
        return Err(CompressImageError::InputTooLarge {
            size: raw_bytes.len(),
            limit: MAX_PROMPT_IMAGE_INPUT_BYTES,
        });
    }

    let dims = ImageReader::new(Cursor::new(&raw_bytes))
        .with_guessed_format()
        .ok()
        .and_then(|r| r.into_dimensions().ok());

    // Pass through untouched only if the bytes are a structurally complete
    // JPEG/PNG/WebP — the formats the API accepts on the wire. Anything
    // else (truncated container, HEIC/PSD/unsniffable bytes) falls through
    // to the re-encode chain, which either emits valid endpoint bytes or
    // fails this call — never embedding a payload that would 400 on this
    // and every following turn.
    let passthrough_sendable = match image::guess_format(&raw_bytes) {
        Ok(
            format
            @ (image::ImageFormat::Jpeg | image::ImageFormat::Png | image::ImageFormat::WebP),
        ) => crate::util::image_validate::format_structurally_complete(format, &raw_bytes),
        _ => false,
    };

    if let Some((w, h)) = dims
        && prompt_image_dimensions_fit(w, h, HIGH_DETAIL_LIMITS)
        && passthrough_sendable
    {
        return Ok((raw_bytes, original_mime));
    }

    let reader = match ImageReader::new(Cursor::new(&raw_bytes)).with_guessed_format() {
        Ok(r) => r,
        Err(_) => {
            tracing::warn!("image format detection failed; cannot compress oversized image");
            return Err(CompressImageError::FormatDetectionFailed);
        }
    };

    if reader.format().is_none() {
        tracing::warn!("image format unknown; cannot compress oversized image");
        return Err(CompressImageError::FormatDetectionFailed);
    }

    if let Ok((w, h)) = reader.into_dimensions()
        && (w as u64) * (h as u64) > MAX_DECODE_PIXELS
    {
        tracing::warn!(
            width = w,
            height = h,
            "image exceeds {MAX_DECODE_PIXELS} px decode limit; cannot compress"
        );
        return Err(CompressImageError::PixelLimitExceeded {
            width: w,
            height: h,
            limit_pixels: MAX_DECODE_PIXELS,
        });
    }

    let img = match ImageReader::new(Cursor::new(&raw_bytes))
        .with_guessed_format()
        .ok()
        .and_then(|r| r.decode().ok())
    {
        Some(img) => img,
        None => {
            tracing::warn!("image decode failed; cannot compress oversized image");
            return Err(CompressImageError::DecodeFailed(
                "pixel decode returned no image".into(),
            ));
        }
    };

    let preferred = preferred_prompt_format(&raw_bytes);
    let (buf, _w, _h, mime) = match resize_and_encode_prompt_image(&img, preferred, HIGH_DETAIL_LIMITS)
    {
        Ok(v) => v,
        Err(ReEncodeError::EncodeFailed | ReEncodeError::CouldNotFit { .. }) => {
            tracing::warn!("image re-encode failed; cannot embed image");
            return Err(CompressImageError::DecodeFailed(
                "prompt image encode failed".into(),
            ));
        }
    };

    Ok((buf, mime.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_noisy_png(width: u32, height: u32) -> Vec<u8> {
        use image::{ImageBuffer, Rgba};
        let img = ImageBuffer::from_fn(width, height, |x, y| {
            let seed = (x as u64).wrapping_mul(6364136223846793005)
                ^ (y as u64).wrapping_mul(1442695040888963407);
            let r = (seed & 0xFF) as u8;
            let g = ((seed >> 8) & 0xFF) as u8;
            let b = ((seed >> 16) & 0xFF) as u8;
            Rgba([r, g, b, 255u8])
        });
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    fn make_small_png(width: u32, height: u32) -> Vec<u8> {
        use image::{ImageBuffer, Rgba};
        let img = ImageBuffer::from_pixel(width, height, Rgba([0u8, 0, 0, 255]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn compress_small_image_returns_unchanged() {
        let png = make_small_png(16, 16);
        let (result, mime) =
            compress_image_for_conversation(png.clone(), "image/png".into()).unwrap();
        assert_eq!(result, png);
        assert_eq!(mime, "image/png");
    }

    /// A truncated small JPEG must not pass through raw; it falls through
    /// to re-encode, which emits structurally complete bytes from the
    /// decodable portion.
    #[test]
    fn compress_truncated_small_jpeg_re_encodes_to_valid_bytes() {
        use image::codecs::jpeg::JpegEncoder;
        use image::{DynamicImage, ImageBuffer, Rgb};
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(200, 150, |x, y| {
            Rgb([(x ^ y) as u8, (x * 3) as u8, (y * 5) as u8])
        });
        let mut jpeg = Vec::new();
        JpegEncoder::new_with_quality(&mut jpeg, 85)
            .encode_image(&DynamicImage::ImageRgb8(img))
            .unwrap();
        jpeg.truncate(jpeg.len() / 2);
        assert!(
            !crate::util::image_validate::jpeg_reaches_eoi(&jpeg),
            "precondition: input is structurally incomplete"
        );
        let (result, _mime) =
            compress_image_for_conversation(jpeg.clone(), "image/jpeg".into()).unwrap();
        assert_ne!(result, jpeg, "raw truncated bytes must not pass through");
        assert!(
            crate::util::image_validate::image_structurally_complete(&result),
            "output must be structurally complete"
        );
    }

    /// Small GIF must become PNG (not pass through as image/gif).
    #[test]
    fn compress_small_gif_becomes_png() {
        use image::{ImageBuffer, Rgba};
        let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(24, 24, Rgba([1u8, 2, 3, 255]));
        let mut gif = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut gif), image::ImageFormat::Gif)
            .unwrap();
        let (result, mime) =
            compress_image_for_conversation(gif, "image/gif".into()).expect("gif compresses");
        assert_eq!(mime, "image/png");
        assert_eq!(
            image::guess_format(&result).unwrap(),
            image::ImageFormat::Png
        );
    }

    #[test]
    fn compress_large_noisy_png_stays_png_and_fits_high() {
        use crate::util::image_compress::{HIGH_DETAIL_LIMITS, prompt_image_dimensions_fit};
        let png = make_noisy_png(2048, 1536);
        let (result, mime) = compress_image_for_conversation(png, "image/png".into()).unwrap();
        assert_eq!(mime, "image/png");
        let (w, h) = image::ImageReader::new(std::io::Cursor::new(&result))
            .with_guessed_format()
            .unwrap()
            .into_dimensions()
            .unwrap();
        assert_eq!((w, h), (1824, 1368));
        assert!(prompt_image_dimensions_fit(w, h, HIGH_DETAIL_LIMITS));
    }

    /// Flat-colour image: huge in pixels, tiny in bytes. Codex high still
    /// downscales (2048×2600 exceeds the side cap).
    #[test]
    fn compress_large_dimensions_small_bytes_downscales() {
        use crate::util::image_compress::{HIGH_DETAIL_LIMITS, prompt_image_dimensions_fit};
        let png = make_small_png(2048, 2600);
        let (result, mime) =
            compress_image_for_conversation(png, "image/png".into()).expect("downscale succeeds");
        assert_eq!(mime, "image/png");
        let (w, h) = image::ImageReader::new(std::io::Cursor::new(&result))
            .with_guessed_format()
            .unwrap()
            .into_dimensions()
            .unwrap();
        // 2048×2600: side clamp then patch budget → 1408×1787.
        assert_eq!((w, h), (1408, 1787));
        assert!(prompt_image_dimensions_fit(w, h, HIGH_DETAIL_LIMITS));
    }

    /// Wide image within Codex high (1600×600) passes through untouched.
    #[test]
    fn compress_wide_image_under_high_budget_passes_through() {
        let png = make_small_png(1600, 600);
        let (result, mime) =
            compress_image_for_conversation(png.clone(), "image/png".into()).unwrap();
        assert_eq!(result, png, "within-budget image must not be re-encoded");
        assert_eq!(mime, "image/png");
    }

    /// 3438×1830 screenshot: Codex high lands on 2048×1090.
    #[test]
    fn compress_screenshot_matches_codex_high() {
        let png = make_small_png(3438, 1830);
        let (result, _mime) =
            compress_image_for_conversation(png, "image/png".into()).expect("downscale succeeds");
        let (w, h) = image::ImageReader::new(std::io::Cursor::new(&result))
            .with_guessed_format()
            .unwrap()
            .into_dimensions()
            .unwrap();
        assert_eq!((w, h), (2048, 1090));
        let r_in = 3438.0 / 1830.0;
        let r_out = w as f64 / h as f64;
        assert!(
            (r_in - r_out).abs() < 0.05,
            "aspect ratio {r_in} -> {r_out} ({w}x{h})"
        );
    }

    /// Regression: a 25 Mpx camera-class photo (cf. a real 5184×3888 iPhone
    /// shot rejected under the old 16 Mpx cap) must compress, not error —
    /// the API accepts up to ~178.9 Mpx and we downscale before the wire.
    #[test]
    fn compress_camera_sized_photo_succeeds() {
        use crate::util::image_compress::{HIGH_DETAIL_LIMITS, prompt_image_dimensions_fit};
        let png = make_noisy_png(5000, 5000);
        let (out, mime) = compress_image_for_conversation(png, "image/png".into())
            .expect("25 Mpx photo must compress");
        assert_eq!(mime, "image/png");
        let (w, h, _) = crate::util::image_validate::validate_image_bytes(&out).unwrap();
        assert_eq!((w, h), (1600, 1600));
        assert!(prompt_image_dimensions_fit(w, h, HIGH_DETAIL_LIMITS));
    }

    /// Above the API's own ceiling the decode is refused (the API would 400
    /// it regardless). SOF dims are patched — encoding a real >178 Mpx
    /// fixture is infeasible.
    #[test]
    fn compress_above_api_ceiling_returns_pixel_limit_exceeded() {
        use image::codecs::jpeg::JpegEncoder;
        use image::{DynamicImage, ImageBuffer, Rgb};
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(64, 64, Rgb([7, 8, 9]));
        let mut jpeg = Vec::new();
        JpegEncoder::new_with_quality(&mut jpeg, 85)
            .encode_image(&DynamicImage::ImageRgb8(img))
            .unwrap();
        let sof = jpeg
            .windows(2)
            .position(|w| w == [0xFF, 0xC0])
            .expect("baseline SOF0 present");
        // 16384 x 16384 = 268 Mpx, above the 178.9 Mpx ceiling.
        jpeg[sof + 5..sof + 9].copy_from_slice(&[0x40, 0x00, 0x40, 0x00]);
        let err = compress_image_for_conversation(jpeg, "image/jpeg".into()).unwrap_err();
        match err {
            CompressImageError::PixelLimitExceeded {
                width,
                height,
                limit_pixels,
            } => {
                assert_eq!((width, height), (16384, 16384));
                assert_eq!(limit_pixels, MAX_DECODE_PIXELS);
            }
            other => panic!("expected PixelLimitExceeded, got {other:?}"),
        }
    }

    #[test]
    fn compress_small_undecodable_fails_closed() {
        let garbage = b"not an image at all".to_vec();
        let err = compress_image_for_conversation(garbage, "image/svg+xml".into())
            .expect_err("unsniffable bytes must not pass through raw");
        assert!(
            matches!(err, CompressImageError::FormatDetectionFailed),
            "got: {err:?}"
        );
    }

    #[test]
    fn compress_within_high_budget_passes_through_regardless_of_bytes() {
        let png = make_noisy_png(1400, 1050);
        let (result, mime) =
            compress_image_for_conversation(png.clone(), "image/png".into()).unwrap();
        assert_eq!(result, png, "1400x1050 fits Codex high; no byte re-encode");
        assert_eq!(mime, "image/png");
    }

    /// All-zero bytes → `FormatDetectionFailed`.
    #[test]
    fn compress_oversized_zero_bytes_returns_format_detection_failed() {
        let bytes = vec![0u8; MAX_IMAGE_PAYLOAD_BYTES + 4096];
        let err = compress_image_for_conversation(bytes, "image/png".into()).unwrap_err();
        assert!(
            matches!(err, CompressImageError::FormatDetectionFailed),
            "got {err:?}"
        );
    }

    /// Valid PNG header + corrupted IDAT → `DecodeFailed`.
    #[test]
    fn compress_oversized_corrupt_png_returns_decode_failed() {
        let mut png = make_noisy_png(1024, 1024);
        let tag = b"IDAT";
        let pos = png.windows(4).position(|w| w == tag).unwrap();
        for i in 0..512 {
            png[pos + 8 + i] ^= 0x5A;
        }
        let err = compress_image_for_conversation(png, "image/png".into()).unwrap_err();
        assert!(
            matches!(err, CompressImageError::DecodeFailed(_)),
            "got {err:?}"
        );
    }

    /// Tiny caps are no longer a prompt-image failure mode. Keep the
    /// `PayloadCapExceeded` Display pin for callers that still match the variant.
    #[test]
    fn payload_cap_exceeded_display_string_pinned() {
        let cap_err = CompressImageError::PayloadCapExceeded(768 * 1024);
        assert!(
            cap_err.to_string().contains("786432"),
            "rendered: {cap_err}"
        );
    }

    #[test]
    fn input_too_large_display_pins_codex_one_gib_cap() {
        use crate::util::image_compress::MAX_PROMPT_IMAGE_INPUT_BYTES;
        let err = CompressImageError::InputTooLarge {
            size: MAX_PROMPT_IMAGE_INPUT_BYTES + 1,
            limit: MAX_PROMPT_IMAGE_INPUT_BYTES,
        };
        let rendered = err.to_string();
        assert!(
            rendered.contains(&(MAX_PROMPT_IMAGE_INPUT_BYTES + 1).to_string()),
            "rendered: {rendered}"
        );
        assert!(
            rendered.contains(&MAX_PROMPT_IMAGE_INPUT_BYTES.to_string()),
            "rendered: {rendered}"
        );
    }
}
