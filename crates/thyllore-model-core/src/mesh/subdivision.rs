use std::collections::HashMap;

pub struct PolygonMesh {
    pub points: Vec<[f32; 3]>,
    pub face_counts: Vec<u32>,
    pub face_indices: Vec<u32>,
}

pub struct SubdividedMesh {
    pub points: Vec<[f32; 3]>,
    pub face_counts: Vec<u32>,
    pub face_indices: Vec<u32>,
    pub weights: Vec<SparseWeights>,
    pub uvs: Vec<[f32; 2]>,
}

pub trait Blendable: Clone {
    fn zero() -> Self;
    fn add_scaled(&mut self, other: &Self, scale: f32);
}

impl Blendable for [f32; 3] {
    fn zero() -> Self {
        [0.0, 0.0, 0.0]
    }
    fn add_scaled(&mut self, other: &Self, scale: f32) {
        self[0] += other[0] * scale;
        self[1] += other[1] * scale;
        self[2] += other[2] * scale;
    }
}

impl Blendable for [f32; 2] {
    fn zero() -> Self {
        [0.0, 0.0]
    }
    fn add_scaled(&mut self, other: &Self, scale: f32) {
        self[0] += other[0] * scale;
        self[1] += other[1] * scale;
    }
}

#[derive(Clone, Default)]
pub struct SparseWeights(pub Vec<(u32, f32)>);

impl Blendable for SparseWeights {
    fn zero() -> Self {
        SparseWeights(Vec::new())
    }
    fn add_scaled(&mut self, other: &Self, scale: f32) {
        for &(bone, weight) in &other.0 {
            let scaled = weight * scale;
            if let Some(entry) = self.0.iter_mut().find(|e| e.0 == bone) {
                entry.1 += scaled;
            } else {
                self.0.push((bone, scaled));
            }
        }
    }
}

impl SparseWeights {
    pub fn into_top4_normalized(mut self) -> ([u32; 4], [f32; 4]) {
        self.0
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut indices = [0u32; 4];
        let mut weights = [0.0f32; 4];
        let mut total = 0.0;
        for (slot, &(bone, weight)) in self.0.iter().take(4).enumerate() {
            indices[slot] = bone;
            weights[slot] = weight;
            total += weight;
        }
        if total > 0.0 {
            for weight in &mut weights {
                *weight /= total;
            }
        }
        (indices, weights)
    }
}

pub fn catmull_clark(
    cage: &PolygonMesh,
    point_weights: &[SparseWeights],
    point_uvs: &[[f32; 2]],
    levels: u32,
) -> SubdividedMesh {
    let mut points = cage.points.clone();
    let mut counts = cage.face_counts.clone();
    let mut indices = cage.face_indices.clone();
    let mut weights = point_weights.to_vec();
    let mut uvs = point_uvs.to_vec();

    for _ in 0..levels {
        let topology = build_step_topology(&counts, &indices, points.len());
        points = blend_attribute(&topology, &points);
        weights = blend_attribute(&topology, &weights);
        uvs = blend_attribute(&topology, &uvs);
        counts = topology.new_face_counts;
        indices = topology.new_face_indices;
    }

    SubdividedMesh {
        points,
        face_counts: counts,
        face_indices: indices,
        weights,
        uvs,
    }
}

pub fn triangulate_subdivided(mesh: &SubdividedMesh) -> Vec<u32> {
    let mut triangles = Vec::new();
    let mut offset = 0usize;

    for &count in &mesh.face_counts {
        let corner_count = count as usize;
        if corner_count >= 3 {
            for corner in 1..corner_count - 1 {
                triangles.push(mesh.face_indices[offset]);
                triangles.push(mesh.face_indices[offset + corner]);
                triangles.push(mesh.face_indices[offset + corner + 1]);
            }
        }
        offset += corner_count;
    }

    triangles
}

struct StepTopology {
    new_face_counts: Vec<u32>,
    new_face_indices: Vec<u32>,
    faces: Vec<Vec<u32>>,
    edges: Vec<(u32, u32)>,
    edge_faces: Vec<Vec<u32>>,
    vertex_faces: Vec<Vec<u32>>,
    vertex_edges: Vec<Vec<u32>>,
    boundary_edge: Vec<bool>,
    boundary_vertex: Vec<bool>,
}

fn build_step_topology(
    face_counts: &[u32],
    face_indices: &[u32],
    point_count: usize,
) -> StepTopology {
    let faces = collect_faces(face_counts, face_indices);

    let mut edge_id: HashMap<(u32, u32), u32> = HashMap::new();
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut edge_faces: Vec<Vec<u32>> = Vec::new();
    let mut vertex_faces: Vec<Vec<u32>> = vec![Vec::new(); point_count];
    let mut vertex_edges: Vec<Vec<u32>> = vec![Vec::new(); point_count];

    for (face_index, corners) in faces.iter().enumerate() {
        let valence = corners.len();
        for corner in 0..valence {
            let a = corners[corner];
            let b = corners[(corner + 1) % valence];
            let key = (a.min(b), a.max(b));

            let edge = *edge_id.entry(key).or_insert_with(|| {
                let id = edges.len() as u32;
                edges.push((a, b));
                edge_faces.push(Vec::new());
                id
            });

            let edge_idx = edge as usize;
            if !edge_faces[edge_idx].contains(&(face_index as u32)) {
                edge_faces[edge_idx].push(face_index as u32);
            }
            if !vertex_edges[a as usize].contains(&edge) {
                vertex_edges[a as usize].push(edge);
            }
            if !vertex_edges[b as usize].contains(&edge) {
                vertex_edges[b as usize].push(edge);
            }
            vertex_faces[a as usize].push(face_index as u32);
        }
    }

    let boundary_edge: Vec<bool> = edge_faces.iter().map(|f| f.len() == 1).collect();
    let boundary_vertex: Vec<bool> = (0..point_count)
        .map(|v| vertex_edges[v].iter().any(|&e| boundary_edge[e as usize]))
        .collect();

    let face_point_base = 0u32;
    let edge_point_base = faces.len() as u32;
    let vertex_point_base = edge_point_base + edges.len() as u32;

    let mut new_face_counts = Vec::new();
    let mut new_face_indices = Vec::new();
    for (face_index, corners) in faces.iter().enumerate() {
        let valence = corners.len();
        let face_point = face_point_base + face_index as u32;
        for corner in 0..valence {
            let current = corners[corner];
            let next = corners[(corner + 1) % valence];
            let previous = corners[(corner + valence - 1) % valence];

            let edge_next = edge_point_base + edge_id[&(current.min(next), current.max(next))];
            let edge_prev =
                edge_point_base + edge_id[&(previous.min(current), previous.max(current))];
            let vertex_point = vertex_point_base + current;

            new_face_indices.extend_from_slice(&[vertex_point, edge_next, face_point, edge_prev]);
            new_face_counts.push(4);
        }
    }

    StepTopology {
        new_face_counts,
        new_face_indices,
        faces,
        edges,
        edge_faces,
        vertex_faces,
        vertex_edges,
        boundary_edge,
        boundary_vertex,
    }
}

fn collect_faces(face_counts: &[u32], face_indices: &[u32]) -> Vec<Vec<u32>> {
    let mut faces = Vec::with_capacity(face_counts.len());
    let mut offset = 0usize;
    for &count in face_counts {
        let valence = count as usize;
        if valence >= 3 && offset + valence <= face_indices.len() {
            faces.push(face_indices[offset..offset + valence].to_vec());
        }
        offset += valence;
    }
    faces
}

fn blend_attribute<A: Blendable>(topology: &StepTopology, source: &[A]) -> Vec<A> {
    let face_points = compute_face_points(topology, source);
    let edge_points = compute_edge_points(topology, source, &face_points);
    let vertex_points = compute_vertex_points(topology, source, &face_points);

    let mut result =
        Vec::with_capacity(face_points.len() + edge_points.len() + vertex_points.len());
    result.extend(face_points);
    result.extend(edge_points);
    result.extend(vertex_points);
    result
}

fn compute_face_points<A: Blendable>(topology: &StepTopology, source: &[A]) -> Vec<A> {
    topology
        .faces
        .iter()
        .map(|corners| {
            let mut accumulator = A::zero();
            let scale = 1.0 / corners.len() as f32;
            for &corner in corners {
                accumulator.add_scaled(&source[corner as usize], scale);
            }
            accumulator
        })
        .collect()
}

fn compute_edge_points<A: Blendable>(
    topology: &StepTopology,
    source: &[A],
    face_points: &[A],
) -> Vec<A> {
    topology
        .edges
        .iter()
        .enumerate()
        .map(|(edge, &(a, b))| {
            let adjacent_faces = &topology.edge_faces[edge];
            let mut accumulator = A::zero();

            if topology.boundary_edge[edge] {
                accumulator.add_scaled(&source[a as usize], 0.5);
                accumulator.add_scaled(&source[b as usize], 0.5);
            } else {
                let divisor = (2 + adjacent_faces.len()) as f32;
                let coefficient = 1.0 / divisor;
                accumulator.add_scaled(&source[a as usize], coefficient);
                accumulator.add_scaled(&source[b as usize], coefficient);
                for &face in adjacent_faces {
                    accumulator.add_scaled(&face_points[face as usize], coefficient);
                }
            }
            accumulator
        })
        .collect()
}

fn compute_vertex_points<A: Blendable>(
    topology: &StepTopology,
    source: &[A],
    face_points: &[A],
) -> Vec<A> {
    (0..source.len())
        .map(|vertex| {
            let incident_edges = &topology.vertex_edges[vertex];
            if incident_edges.is_empty() {
                return source[vertex].clone();
            }

            if topology.boundary_vertex[vertex] {
                blend_boundary_vertex(topology, source, vertex, incident_edges)
            } else {
                blend_interior_vertex(topology, source, face_points, vertex, incident_edges)
            }
        })
        .collect()
}

fn blend_boundary_vertex<A: Blendable>(
    topology: &StepTopology,
    source: &[A],
    vertex: usize,
    incident_edges: &[u32],
) -> A {
    let mut neighbors = Vec::new();
    for &edge in incident_edges {
        if topology.boundary_edge[edge as usize] {
            let (a, b) = topology.edges[edge as usize];
            neighbors.push(if a as usize == vertex { b } else { a });
        }
    }

    if neighbors.len() != 2 {
        return source[vertex].clone();
    }

    let mut accumulator = A::zero();
    accumulator.add_scaled(&source[vertex], 0.75);
    accumulator.add_scaled(&source[neighbors[0] as usize], 0.125);
    accumulator.add_scaled(&source[neighbors[1] as usize], 0.125);
    accumulator
}

fn blend_interior_vertex<A: Blendable>(
    topology: &StepTopology,
    source: &[A],
    face_points: &[A],
    vertex: usize,
    incident_edges: &[u32],
) -> A {
    let valence = incident_edges.len() as f32;
    let mut accumulator = A::zero();

    let adjacent_faces = &topology.vertex_faces[vertex];
    if !adjacent_faces.is_empty() {
        let face_scale = 1.0 / adjacent_faces.len() as f32 / valence;
        for &face in adjacent_faces {
            accumulator.add_scaled(&face_points[face as usize], face_scale);
        }
    }

    let edge_scale = 2.0 / valence / valence;
    for &edge in incident_edges {
        let (a, b) = topology.edges[edge as usize];
        let other = if a as usize == vertex { b } else { a };
        accumulator.add_scaled(&source[vertex], edge_scale * 0.5);
        accumulator.add_scaled(&source[other as usize], edge_scale * 0.5);
    }

    accumulator.add_scaled(&source[vertex], (valence - 3.0) / valence);
    accumulator
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_quad() -> PolygonMesh {
        PolygonMesh {
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            face_counts: vec![4],
            face_indices: vec![0, 1, 2, 3],
        }
    }

    #[test]
    fn single_quad_level_one_point_count() {
        let cage = unit_quad();
        let result = catmull_clark(
            &cage,
            &vec![SparseWeights::default(); 4],
            &vec![[0.0, 0.0]; 4],
            1,
        );

        assert_eq!(result.points.len(), 9);
        assert_eq!(result.face_counts.len(), 4);
        assert!(result.face_counts.iter().all(|&c| c == 4));
    }

    #[test]
    fn single_quad_face_point_is_center() {
        let cage = unit_quad();
        let result = catmull_clark(
            &cage,
            &vec![SparseWeights::default(); 4],
            &vec![[0.0, 0.0]; 4],
            1,
        );

        let face_point = result.points[0];
        assert!((face_point[0] - 0.5).abs() < 1e-5);
        assert!((face_point[1] - 0.5).abs() < 1e-5);
        assert!((face_point[2]).abs() < 1e-5);
    }

    #[test]
    fn weights_remain_normalized() {
        let cage = unit_quad();
        let point_weights = vec![
            SparseWeights(vec![(0, 1.0)]),
            SparseWeights(vec![(1, 1.0)]),
            SparseWeights(vec![(0, 0.5), (1, 0.5)]),
            SparseWeights(vec![(1, 1.0)]),
        ];
        let result = catmull_clark(&cage, &point_weights, &vec![[0.0, 0.0]; 4], 2);

        for weight in &result.weights {
            let total: f32 = weight.0.iter().map(|w| w.1).sum();
            assert!((total - 1.0).abs() < 1e-4, "weight sum drifted: {}", total);
        }
    }

    #[test]
    fn subdivided_points_stay_in_unit_bounds() {
        let cage = unit_quad();
        let result = catmull_clark(
            &cage,
            &vec![SparseWeights::default(); 4],
            &vec![[0.0, 0.0]; 4],
            2,
        );

        for point in &result.points {
            assert!(point[0] >= -1e-4 && point[0] <= 1.0 + 1e-4);
            assert!(point[1] >= -1e-4 && point[1] <= 1.0 + 1e-4);
        }
    }

    #[test]
    fn isolated_point_survives() {
        let cage = PolygonMesh {
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 5.0, 5.0],
            ],
            face_counts: vec![4],
            face_indices: vec![0, 1, 2, 3],
        };
        let result = catmull_clark(
            &cage,
            &vec![SparseWeights::default(); 5],
            &vec![[0.0, 0.0]; 5],
            1,
        );

        let isolated = result.points.iter().any(|p| {
            (p[0] - 5.0).abs() < 1e-4 && (p[1] - 5.0).abs() < 1e-4 && (p[2] - 5.0).abs() < 1e-4
        });
        assert!(isolated, "isolated point should pass through unchanged");
    }
}
