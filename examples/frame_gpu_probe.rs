use macroquad::{miniquad, prelude::*, telemetry};
use std::collections::HashSet;

fn main() {
    macroquad::Window::from_config(
        miniquad::conf::Conf {
            headless: true,
            icon: None,
            window_title: "Frame GPU probe".to_owned(),
            ..Default::default()
        },
        probe(),
    );
}

async fn probe() {
    use miniquad::native::gl;

    let string = |name| unsafe {
        let value = gl::glGetString(name);
        assert!(!value.is_null());
        std::ffi::CStr::from_ptr(value as _)
            .to_string_lossy()
            .into_owned()
    };
    let vendor = string(gl::GL_VENDOR);
    let renderer = string(0x1F01);
    let version = string(gl::GL_VERSION);
    println!("GL_VENDOR={vendor} GL_RENDERER={renderer} GL_VERSION={version}");
    if std::env::var("PHIRA_REQUIRE_UHD630").as_deref() == Ok("1") {
        assert!(vendor.to_ascii_lowercase().contains("intel"), "{}", vendor);
        assert!(
            renderer.contains("UHD") && renderer.contains("630"),
            "{}",
            renderer
        );
    }
    assert!(telemetry::frame_gpu_sampling_stats().supported);

    let mut seen_frames = HashSet::new();
    let mut sampled_frames = HashSet::new();
    telemetry::set_frame_gpu_sampling(true);
    for _ in 0..240 {
        let frame_id = telemetry::current_frame_id();
        for sample in telemetry::take_frame_gpu_samples() {
            assert!(sample.submission_frame < frame_id);
            assert!(seen_frames.contains(&sample.submission_frame));
            assert!(sampled_frames.insert(sample.submission_frame));
            assert!(sample.elapsed_ns > 0);
            println!(
                "frame={} elapsed_ns={}",
                sample.submission_frame, sample.elapsed_ns
            );
        }
        assert!(seen_frames.insert(frame_id));
        clear_background(DARKGRAY);
        for index in 0..64 {
            draw_rectangle(index as f32 * 4.0, index as f32 * 2.0, 120.0, 80.0, GREEN);
        }
        if sampled_frames.len() >= 8 {
            break;
        }
        next_frame().await;
    }
    assert!(
        sampled_frames.len() >= 8,
        "fewer than eight completed samples"
    );
    telemetry::set_frame_gpu_sampling(false);
    next_frame().await;
    next_frame().await;
    let stats = telemetry::frame_gpu_sampling_stats();
    assert!(!stats.enabled);
    assert_eq!(stats.pending, 0);
    println!("PASS samples={} stats={stats:?}", sampled_frames.len());
}
