use crate::{get_context, get_quad_context, time::get_time};

use std::collections::HashMap;

mod frame_gpu;
pub use frame_gpu::{FrameGpuSample, FrameGpuSamplingStats};

static mut FRAME_GPU: Option<frame_gpu::FrameGpuSampler> = None;

fn frame_gpu_sampler() -> &'static mut frame_gpu::FrameGpuSampler {
    unsafe { FRAME_GPU.get_or_insert_with(frame_gpu::FrameGpuSampler::new) }
}

/// Opt in to whole-draw GPU timing on the rendering thread. Applied at the next
/// draw frame's start, so disabling during a query never deletes an active timer.
/// Disable and allow one further draw before destroying the GL context to release
/// query objects. Already completed samples remain available after disabling.
pub fn set_frame_gpu_sampling(enabled: bool) {
    frame_gpu_sampler().set_enabled(enabled);
}

/// Drain completed samples. Results arrive asynchronously; missing frames are
/// not represented by zero durations. Only this drain allocates an output Vec.
pub fn take_frame_gpu_samples() -> Vec<FrameGpuSample> {
    frame_gpu_sampler().take_samples()
}

pub fn frame_gpu_sampling_stats() -> FrameGpuSamplingStats {
    frame_gpu_sampler().stats()
}

/// The current actual draw frame, incremented before the user future is polled,
/// even when sampling is disabled. Zero means no draw frame has started yet.
/// Match this ID to FrameGpuSample::submission_frame, not the retrieval frame.
pub fn current_frame_id() -> u64 {
    frame_gpu_sampler().stats().frame_id
}

pub(crate) fn gpu_frame_begin() {
    let legacy_active = get_profiler().queries.values().any(|query| query.in_progress);
    frame_gpu_sampler().begin(legacy_active);
}

pub(crate) fn gpu_frame_end() {
    frame_gpu_sampler().end();
}

static mut PROFILER: Option<Profiler> = None;

fn get_profiler() -> &'static mut Profiler {
    unsafe {
        PROFILER.get_or_insert_with(|| Profiler {
            frame: Frame::new(),
            queries: HashMap::new(),
            active_query: None,
            prev_frame: Frame::new(),
            enabled: false,
            enable_request: None,
            capture_request: false,
            capture: false,
            drawcalls: vec![],
            strings: vec![],
        })
    }
}

#[derive(Debug, Clone)]
pub struct Zone {
    pub name: String,
    pub start_time: f64,
    pub duration: f64,
    pub children: Vec<Zone>,

    parent: *mut Zone,
}

impl Zone {
    fn clone(&self, parent: *mut Zone) -> Zone {
        Zone {
            name: self.name.clone(),
            start_time: self.start_time,
            duration: self.duration,
            children: self
                .children
                .iter()
                .map(|zone| zone.clone(self as *const _ as *mut _))
                .collect(),
            parent,
        }
    }
}

pub struct ZoneGuard {
    _marker: (),
}

impl ZoneGuard {
    pub fn new(name: &str) -> ZoneGuard {
        begin_zone(name);
        ZoneGuard { _marker: () }
    }
}

impl Drop for ZoneGuard {
    fn drop(&mut self) {
        end_zone();
    }
}

pub fn enable() {
    get_profiler().enable_request = Some(true);
}

pub fn disable() {
    get_profiler().enable_request = Some(false);
}

pub fn begin_zone(name: &str) {
    if get_profiler().enabled {
        get_profiler().begin_zone(name);
    }
}

pub fn end_zone() {
    if get_profiler().enabled {
        get_profiler().end_zone();
    }
}

pub fn begin_gpu_query(name: &str) {
    get_profiler().begin_gpu_query(name);
}

pub fn end_gpu_query() {
    get_profiler().end_gpu_query();
}

/// Workaround to stop gl capture on debug rendering
#[doc(hidden)]
pub fn pause_gl_capture() {
    if get_profiler().capture {
        crate::get_context().gl.capture(false);
    }
}

/// Workaround to stop gl capture on debug rendering
pub fn resume_gl_capture() {
    if get_profiler().capture {
        crate::get_context().gl.capture(false);
    }
}

pub(crate) fn reset() {
    let profiler = get_profiler();

    assert!(
        get_profiler().frame.active_zone.is_null(),
        "New frame started with unpaired begin/end zones."
    );

    profiler.frame.full_frame_time = crate::time::get_frame_time();

    std::mem::swap(&mut profiler.prev_frame, &mut profiler.frame);
    profiler.frame = Frame::new();

    if let Some(enable) = profiler.enable_request.take() {
        profiler.enabled = enable;
    }

    if profiler.capture {
        profiler.capture = false;
        crate::get_context().gl.capture(false);
    }

    if profiler.capture_request {
        profiler.drawcalls.clear();
        profiler.capture = true;
        crate::get_context().gl.capture(true);
        profiler.capture_request = false;
    }
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub full_frame_time: f32,
    pub zones: Vec<Zone>,
    active_zone: *mut Zone,
}

impl Frame {
    fn new() -> Frame {
        Frame {
            full_frame_time: 0.0,
            zones: vec![],
            active_zone: std::ptr::null_mut(),
        }
    }

    pub fn try_clone(&self) -> Option<Frame> {
        if self.active_zone.is_null() == false {
            return None;
        }

        Some(Frame {
            full_frame_time: self.full_frame_time,
            zones: self
                .zones
                .iter()
                .map(|zone| zone.clone(std::ptr::null_mut()))
                .collect(),
            active_zone: std::ptr::null_mut(),
        })
    }
}

pub fn frame() -> Frame {
    get_profiler().prev_frame.clone()
}

pub fn gpu_queries() -> Vec<(String, u64)> {
    get_profiler()
        .queries
        .iter()
        .map(|(name, query)| (name.to_owned(), query.value))
        .collect()
}

pub fn sample_gpu_queries() {
    for query in &mut get_profiler().queries {
        query.1.force_resume = true;
    }
}

struct Profiler {
    frame: Frame,
    prev_frame: Frame,
    queries: HashMap<String, GpuQuery>,
    active_query: Option<String>,
    enabled: bool,
    capture_request: bool,
    capture: bool,
    enable_request: Option<bool>,
    drawcalls: Vec<DrawCallTelemetry>,
    strings: Vec<String>,
}

impl Profiler {
    fn begin_gpu_query(&mut self, name: &str) {
        assert!(
            self.active_query.is_none(),
            "Only one active query is allowed by OpenGL"
        );

        let name = name.to_string();
        let query = self
            .queries
            .entry(name.clone())
            .or_insert_with(|| GpuQuery {
                query: miniquad::graphics::ElapsedQuery::new(),
                in_progress: false,
                value: 0,
                force_resume: false,
            });
        self.active_query = Some(name);
        if query.force_resume && !frame_gpu_sampler().stats().enabled {
            query.in_progress = true;
            query.query.begin_query();
        }
    }

    fn end_gpu_query(&mut self) {
        let name = self
            .active_query
            .take()
            .expect("End query without begin query");
        let mut query = self.queries.get_mut(&name).unwrap();
        if query.in_progress {
            query.force_resume = false;
            query.in_progress = false;
            query.query.end_query();
        }
        if query.query.is_available() {
            query.value = query.query.get_result();
        }
    }

    fn begin_zone(&mut self, name: &str) {
        let zones = if self.frame.active_zone.is_null() {
            &mut self.frame.zones
        } else {
            unsafe { &mut (&mut *self.frame.active_zone).children }
        };

        zones.push(Zone {
            name: name.to_string(),
            start_time: get_time(),
            duration: 0.0,
            parent: self.frame.active_zone,
            children: vec![],
        });
        self.frame.active_zone = zones.last_mut().unwrap() as _;
    }

    fn end_zone(&mut self) {
        assert!(
            self.frame.active_zone.is_null() == false,
            "end_zone called without begin_zone"
        );

        let start_time = unsafe { (&mut *self.frame.active_zone).start_time };
        let duration = get_time() - start_time;

        unsafe { (&mut *self.frame.active_zone).duration = duration };
        self.frame.active_zone = unsafe { (&mut *self.frame.active_zone).parent };
    }
}

pub struct GpuQuery {
    pub query: miniquad::graphics::ElapsedQuery,
    pub in_progress: bool,
    pub value: u64,
    pub force_resume: bool,
}

pub fn scene_allocated_memory() -> usize {
    use crate::experimental::scene;

    scene::allocated_memory()
}

/// ```skip
/// {
///    let _t = telemetry::LogTimeGuard::new("Atlas build time");
///     mq::texture::build_textures_atlas();
/// }
/// ```
/// Will add "Time query: Atlas build time, 0.5s" string to
/// `telemetry::strings()`
pub struct LogTimeGuard<'a> {
    name: &'a str,
    start_time: f64,
}

impl<'a> LogTimeGuard<'a> {
    pub fn new(name: &'a str) -> LogTimeGuard {
        LogTimeGuard {
            name,
            start_time: get_time(),
        }
    }
}

impl<'a> Drop for LogTimeGuard<'a> {
    fn drop(&mut self) {
        log_string(&format!(
            "Time query: {}, {:.1}s",
            self.name,
            get_time() - self.start_time
        ))
    }
}

pub fn log_string(string: &str) {
    get_profiler().strings.push(string.to_owned());
}

pub fn drawcalls() -> Vec<DrawCallTelemetry> {
    get_profiler().drawcalls.clone()
}

pub fn strings() -> Vec<String> {
    get_profiler().strings.clone()
}

/// Note that coroutines pre-allocate 56Kb, so 56000 as a result doesnt mean a leak
pub fn coroutines_allocated_memory() -> usize {
    get_context().coroutines_context.allocated_memory()
}

pub fn active_coroutines_count() -> usize {
    get_context().coroutines_context.active_coroutines_count()
}

pub fn capture_frame() {
    get_profiler().capture_request = true;
}

#[derive(Clone, Debug)]
pub struct DrawCallTelemetry {
    pub indices_count: usize,
    pub texture: miniquad::Texture,
}

pub(crate) fn track_drawcall(
    pipeline: &miniquad::Pipeline,
    bindings: &miniquad::Bindings,
    vertex_offset: usize,
    indices_start: usize,
    indices_count: usize,
) {
    let texture = miniquad::Texture::new_render_texture(
        get_quad_context(),
        miniquad::TextureParams {
            width: 128,
            height: 128,
            ..Default::default()
        },
    );

    let pass = Some(miniquad::RenderPass::new(get_quad_context(), texture, None));
    get_quad_context().begin_pass(pass, miniquad::PassAction::clear_color(0.4, 0.8, 0.4, 1.));
    get_quad_context().apply_pipeline(pipeline);
    get_quad_context().apply_bindings_with_offsets(bindings, &[vertex_offset]);
    get_quad_context().draw(indices_start as _, indices_count as _, 1);
    get_quad_context().end_render_pass();

    get_profiler().drawcalls.push(DrawCallTelemetry {
        indices_count,
        texture,
    });
}
