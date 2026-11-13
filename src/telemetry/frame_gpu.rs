use miniquad::graphics::ElapsedQuery;
use std::collections::VecDeque;

const QUERY_COUNT: usize = 8;
const RESULT_COUNT: usize = 256;

/// A completed GPU interval, identified by the draw frame that submitted it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameGpuSample {
    pub submission_frame: u64,
    pub elapsed_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameGpuSamplingStats {
    /// Effective state; requests are applied at the next draw frame's start.
    pub enabled: bool,
    /// False until the first draw frame has checked the current context.
    pub supported: bool,
    pub pending: usize,
    /// Skipped submissions, discarded pending queries, and overwritten ready samples.
    pub dropped: u64,
    pub frame_id: u64,
}

pub(crate) trait Query {
    fn new() -> Self;
    fn supported() -> bool;
    fn begin(&mut self);
    fn end(&mut self);
    fn result(&self) -> Option<u64>;
    fn delete(&mut self);
}

impl Query for ElapsedQuery {
    fn new() -> Self {
        Self::new()
    }
    fn supported() -> bool {
        Self::is_supported()
    }
    fn begin(&mut self) {
        self.begin_query();
    }
    fn end(&mut self) {
        self.end_query();
    }
    fn result(&self) -> Option<u64> {
        self.try_get_result()
    }
    fn delete(&mut self) {
        self.delete();
    }
}

pub(crate) struct FrameGpuSampler<Q = ElapsedQuery> {
    queries: [Q; QUERY_COUNT],
    pending: VecDeque<(usize, u64)>,
    ready: VecDeque<FrameGpuSample>,
    active: Option<usize>,
    enabled: bool,
    request: Option<bool>,
    supported: Option<bool>,
    frame_id: u64,
    dropped: u64,
}

impl<Q: Query> FrameGpuSampler<Q> {
    pub(crate) fn new() -> Self {
        Self {
            queries: std::array::from_fn(|_| Q::new()),
            pending: VecDeque::with_capacity(QUERY_COUNT),
            ready: VecDeque::with_capacity(RESULT_COUNT),
            active: None,
            enabled: false,
            request: None,
            supported: None,
            frame_id: 0,
            dropped: 0,
        }
    }

    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        self.request = Some(enabled);
    }

    pub(crate) fn stats(&self) -> FrameGpuSamplingStats {
        FrameGpuSamplingStats {
            enabled: self.enabled,
            supported: self.supported.unwrap_or(false),
            pending: self.pending.len(),
            dropped: self.dropped,
            frame_id: self.frame_id,
        }
    }

    pub(crate) fn take_samples(&mut self) -> Vec<FrameGpuSample> {
        self.ready.drain(..).collect()
    }

    pub(crate) fn begin(&mut self, legacy_active: bool) {
        self.frame_id = self
            .frame_id
            .checked_add(1)
            .expect("draw frame ID exhausted");
        let supported = *self.supported.get_or_insert_with(Q::supported);
        if let Some(enabled) = self.request.take() {
            self.enabled = enabled;
            if !enabled {
                self.cleanup();
            }
        }
        if !self.enabled || !supported {
            return;
        }

        for _ in 0..self.pending.len() {
            let (slot, submission_frame) = self.pending.pop_front().unwrap();
            if let Some(elapsed_ns) = self.queries[slot].result() {
                if self.ready.len() == RESULT_COUNT {
                    self.ready.pop_front();
                    self.dropped += 1;
                }
                self.ready.push_back(FrameGpuSample {
                    submission_frame,
                    elapsed_ns,
                });
            } else {
                self.pending.push_back((slot, submission_frame));
            }
        }

        if !legacy_active {
            if let Some(slot) = (0..QUERY_COUNT)
                .find(|slot| !self.pending.iter().any(|(pending, _)| pending == slot))
            {
                self.queries[slot].begin();
                self.active = Some(slot);
                return;
            }
        }
        self.dropped += 1;
    }

    pub(crate) fn end(&mut self) {
        if let Some(slot) = self.active.take() {
            self.queries[slot].end();
            self.pending.push_back((slot, self.frame_id));
        }
    }

    fn cleanup(&mut self) {
        assert!(self.active.is_none());
        self.dropped += self.pending.len() as u64;
        self.pending.clear();
        for query in &mut self.queries {
            query.delete();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct FakeQuery {
        ready: Cell<bool>,
        polls: Cell<usize>,
        begins: usize,
        ends: usize,
        deletes: usize,
    }

    impl Query for FakeQuery {
        fn new() -> Self {
            Self {
                ready: Cell::new(false),
                polls: Cell::new(0),
                begins: 0,
                ends: 0,
                deletes: 0,
            }
        }
        fn supported() -> bool {
            true
        }
        fn begin(&mut self) {
            self.begins += 1;
            self.ready.set(false);
        }
        fn end(&mut self) {
            self.ends += 1;
        }
        fn result(&self) -> Option<u64> {
            self.polls.set(self.polls.get() + 1);
            self.ready.get().then_some(42)
        }
        fn delete(&mut self) {
            self.deletes += 1;
        }
    }

    #[test]
    fn bounded_pending_polls_once_and_preserves_submission_id() {
        let mut sampler = FrameGpuSampler::<FakeQuery>::new();
        sampler.begin(false);
        sampler.end();
        assert_eq!(sampler.stats().frame_id, 1);
        sampler.set_enabled(true);
        for _ in 0..QUERY_COUNT {
            sampler.begin(false);
            sampler.end();
        }
        let polls: Vec<_> = sampler.queries.iter().map(|q| q.polls.get()).collect();
        sampler.begin(false);
        sampler.end();
        assert_eq!(sampler.pending.len(), QUERY_COUNT);
        assert_eq!(sampler.dropped, 1);
        for (q, polls) in sampler.queries.iter().zip(polls) {
            assert_eq!(q.polls.get(), polls + 1);
            assert_eq!(q.begins, 1);
        }
        sampler.queries[0].ready.set(true);
        sampler.begin(false);
        sampler.end();
        assert_eq!(
            sampler.take_samples(),
            vec![FrameGpuSample {
                submission_frame: 2,
                elapsed_ns: 42
            }]
        );
        assert_eq!(sampler.queries[0].begins, 2);
    }

    #[test]
    fn disable_during_active_frame_is_deferred_and_keeps_ready_samples() {
        let mut sampler = FrameGpuSampler::<FakeQuery>::new();
        sampler.set_enabled(true);
        sampler.begin(false);
        sampler.set_enabled(false);
        assert!(sampler.stats().enabled);
        assert_eq!(sampler.queries[0].deletes, 0);
        sampler.end();
        assert_eq!(sampler.queries[0].ends, 1);
        sampler.ready.push_back(FrameGpuSample {
            submission_frame: 0,
            elapsed_ns: 9,
        });
        sampler.begin(false);
        sampler.end();
        assert!(!sampler.stats().enabled);
        assert!(sampler.pending.is_empty());
        assert_eq!(sampler.dropped, 1);
        assert_eq!(sampler.queries[0].deletes, 1);
        assert_eq!(sampler.take_samples().len(), 1);
    }

    #[test]
    fn unsupported_conflicts_and_ready_overflow_are_bounded() {
        let mut sampler = FrameGpuSampler::<FakeQuery>::new();
        sampler.supported = Some(false);
        sampler.set_enabled(true);
        sampler.begin(false);
        sampler.end();
        assert!(!sampler.stats().supported);
        assert_eq!(sampler.queries[0].begins, 0);
        assert!(sampler.take_samples().is_empty());
        sampler.supported = Some(true);
        sampler.begin(true);
        sampler.end();
        assert_eq!(sampler.dropped, 1);
        assert_eq!(sampler.queries[0].begins, 0);
        for _ in 0..RESULT_COUNT + 10 {
            for query in &sampler.queries {
                query.ready.set(true);
            }
            sampler.begin(false);
            sampler.end();
        }
        assert_eq!(sampler.ready.len(), RESULT_COUNT);
        assert_eq!(sampler.dropped, 10);
        assert_eq!(sampler.pending.len(), 1);
    }
}
