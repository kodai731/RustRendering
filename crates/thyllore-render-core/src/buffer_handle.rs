#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u32);

impl Default for BufferHandle {
    fn default() -> Self {
        Self::INVALID
    }
}

impl BufferHandle {
    pub const INVALID: BufferHandle = BufferHandle(u32::MAX);

    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }

    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VertexBufferHandle(pub BufferHandle);

impl VertexBufferHandle {
    pub const INVALID: VertexBufferHandle = VertexBufferHandle(BufferHandle::INVALID);

    pub fn new(id: u32) -> Self {
        Self(BufferHandle::new(id))
    }

    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }

    pub fn index(&self) -> usize {
        self.0.index()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct IndexBufferHandle(pub BufferHandle);

impl IndexBufferHandle {
    pub const INVALID: IndexBufferHandle = IndexBufferHandle(BufferHandle::INVALID);

    pub fn new(id: u32) -> Self {
        Self(BufferHandle::new(id))
    }

    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }

    pub fn index(&self) -> usize {
        self.0.index()
    }
}
