use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PassStage {
    Lighting,
    Effect,
    PostProcess,
    Final,
}

pub trait RenderPassNode: Sync {
    fn name(&self) -> &'static str;
    fn stage(&self) -> PassStage;
    unsafe fn record(
        &self,
        app: &App,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        frame_slot: usize,
    ) -> Result<()>;
}

#[derive(Default)]
pub struct PassGraph {
    nodes: Vec<&'static dyn RenderPassNode>,
}

impl PassGraph {
    pub fn register(&mut self, node: &'static dyn RenderPassNode) {
        self.nodes.retain(|existing| existing.name() != node.name());
        self.nodes.push(node);
        self.nodes.sort_by_key(|node| node.stage());
    }

    pub fn register_all(&mut self, nodes: &[&'static dyn RenderPassNode]) {
        for node in nodes {
            self.register(*node);
        }
    }

    pub fn nodes(&self) -> Vec<&'static dyn RenderPassNode> {
        self.nodes.clone()
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.nodes.iter().map(|node| node.name()).collect()
    }
}

impl std::fmt::Debug for PassGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassGraph")
            .field("nodes", &self.names())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubNode {
        name: &'static str,
        stage: PassStage,
    }

    impl RenderPassNode for StubNode {
        fn name(&self) -> &'static str {
            self.name
        }

        fn stage(&self) -> PassStage {
            self.stage
        }

        unsafe fn record(&self, _: &App, _: vk::CommandBuffer, _: usize, _: usize) -> Result<()> {
            Ok(())
        }
    }

    static TONEMAP: StubNode = StubNode {
        name: "tonemap",
        stage: PassStage::Final,
    };
    static COMPOSITE: StubNode = StubNode {
        name: "composite",
        stage: PassStage::Lighting,
    };
    static WATER: StubNode = StubNode {
        name: "water",
        stage: PassStage::Effect,
    };
    static FLAME: StubNode = StubNode {
        name: "flame",
        stage: PassStage::Effect,
    };

    #[test]
    fn orders_by_stage_then_registration() {
        let mut graph = PassGraph::default();
        graph.register_all(&[&TONEMAP, &COMPOSITE, &WATER, &FLAME]);
        assert_eq!(
            graph.names(),
            vec!["composite", "water", "flame", "tonemap"]
        );
    }

    #[test]
    fn re_registering_same_name_replaces_and_moves_to_end_of_stage() {
        let mut graph = PassGraph::default();
        graph.register_all(&[&WATER, &FLAME]);
        graph.register(&WATER);
        assert_eq!(graph.names(), vec!["flame", "water"]);
    }
}
