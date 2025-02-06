use std::{
    cmp::Ordering,
    collections::{HashSet, VecDeque},
};

#[derive(Clone, Eq, PartialEq, Hash)]
struct State {
    board: [[u8; 3]; 3],
    zero: (u8, u8),
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum Move {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Eq, PartialEq)]
struct Node {
    state: State,
    cost: u32,
    heuristic: u32,
    moves: Vec<Move>,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        (other.cost + other.heuristic).cmp(&(self.cost + self.heuristic))
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl State {
    fn new(board: [[u8; 3]; 3]) -> Self {
        let mut zero = (0, 0);
        for (i, row) in board.iter().enumerate() {
            for (j, cell) in row.iter().enumerate() {
                if *cell == 0 {
                    zero = (i as u8, j as u8);
                }
            }
        }
        State { board, zero }
    }

    fn get_neighbors(&self) -> Vec<(State, Move)> {
        let mut neighbors = Vec::new();
        let (i, j) = self.zero;
        if i > 0 {
            let mut new_board = self.board;
            new_board[i as usize][j as usize] = new_board[i as usize - 1][j as usize];
            new_board[i as usize - 1][j as usize] = 0;
            neighbors.push((
                State {
                    board: new_board,
                    zero: (i - 1, j),
                },
                Move::Up,
            ));
        }
        if i < 2 {
            let mut new_board = self.board;
            new_board[i as usize][j as usize] = new_board[i as usize + 1][j as usize];
            new_board[i as usize + 1][j as usize] = 0;
            neighbors.push((
                State {
                    board: new_board,
                    zero: (i + 1, j),
                },
                Move::Down,
            ));
        }
        if j > 0 {
            let mut new_board = self.board;
            new_board[i as usize][j as usize] = new_board[i as usize][j as usize - 1];
            new_board[i as usize][j as usize - 1] = 0;
            neighbors.push((
                State {
                    board: new_board,
                    zero: (i, j - 1),
                },
                Move::Left,
            ));
        }
        if j < 2 {
            let mut new_board = self.board;
            new_board[i as usize][j as usize] = new_board[i as usize][j as usize + 1];
            new_board[i as usize][j as usize + 1] = 0;
            neighbors.push((
                State {
                    board: new_board,
                    zero: (i, j + 1),
                },
                Move::Right,
            ));
        }
        neighbors
    }

    fn is_goal(&self) -> bool {
        let goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]];
        self.board == goal
    }
}

fn manhattan_distance(board: [[u8; 3]; 3]) -> u32 {
    let mut distance = 0;
    for (i, row) in board.iter().enumerate() {
        for (j, cell) in row.iter().enumerate() {
            if *cell != 0 {
                let target_i = (*cell - 1) / 3;
                let target_j = (*cell - 1) % 3;
                distance += (target_i as i32 - i as i32).unsigned_abs();
                distance += (target_j as i32 - j as i32).unsigned_abs();
            }
        }
    }
    distance
}

fn solve_bfs(board: [[u8; 3]; 3]) -> (State, Vec<Move>, u32) {
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    let mut nodes_visited = 0;

    let start = State::new(board);
    queue.push_back((start.clone(), Vec::new()));
    visited.insert(start);

    while let Some((state, moves)) = queue.pop_front() {
        nodes_visited += 1;
        if state.is_goal() {
            return (state, moves, nodes_visited);
        }
        for (neighbor, mv) in state.get_neighbors() {
            if !visited.contains(&neighbor) {
                let mut new_moves = moves.clone();
                new_moves.push(mv);
                queue.push_back((neighbor.clone(), new_moves));
                visited.insert(neighbor);
            }
        }
    }

    panic!("No solution found");
}

fn solve_dfs(board: [[u8; 3]; 3]) -> (State, Vec<Move>, u32) {
    let mut stack = Vec::new();
    let mut visited = HashSet::new();
    let mut nodes_visited = 0;

    let start = State::new(board);
    stack.push((start.clone(), Vec::new()));
    visited.insert(start);

    while let Some((state, moves)) = stack.pop() {
        nodes_visited += 1;
        if state.is_goal() {
            return (state, moves, nodes_visited);
        }
        for (neighbor, mv) in state.get_neighbors() {
            if !visited.contains(&neighbor) {
                let mut new_moves = moves.clone();
                new_moves.push(mv);
                stack.push((neighbor.clone(), new_moves));
                visited.insert(neighbor);
            }
        }
    }

    panic!("No solution found");
}

fn solve_astar(board: [[u8; 3]; 3]) -> (State, Vec<Move>, u32) {
    let mut heap = std::collections::BinaryHeap::new();
    let mut visited = HashSet::new();
    let mut nodes_visited = 0;

    let start = State::new(board);
    heap.push(Node {
        state: start.clone(),
        cost: 0,
        heuristic: manhattan_distance(start.board),
        moves: Vec::new(),
    });
    visited.insert(start);

    while let Some(Node {
        state,
        cost,
        heuristic: _,
        moves,
    }) = heap.pop()
    {
        nodes_visited += 1;
        if state.is_goal() {
            return (state, moves, nodes_visited);
        }
        for (neighbor, mv) in state.get_neighbors() {
            if !visited.contains(&neighbor) {
                let mut new_moves = moves.clone();
                new_moves.push(mv);
                heap.push(Node {
                    state: neighbor.clone(),
                    cost: cost + 1,
                    heuristic: manhattan_distance(neighbor.board),
                    moves: new_moves,
                });
                visited.insert(neighbor);
            }
        }
    }

    panic!("No solution found");
}

fn is_solvable(board: [[u8; 3]; 3]) -> bool {
    let mut inversions = 0;
    let mut flat_board = Vec::new();
    for row in board.iter() {
        for cell in row.iter() {
            flat_board.push(*cell);
        }
    }
    for i in 0..8 {
        for j in i + 1..9 {
            if flat_board[i] != 0 && flat_board[j] != 0 && flat_board[i] > flat_board[j] {
                inversions += 1;
            }
        }
    }
    inversions % 2 == 0
}

fn main() {
    let board = [[5, 1, 4], [3, 8, 2], [6, 7, 0]];

    if !is_solvable(board) {
        println!("The puzzle is not solvable");
        return;
    }

    let (_, _, _) = solve_astar(board);
}
