use std::{
    cell::Cell,
    cmp::Ordering,
    collections::{HashSet, VecDeque},
    sync::Arc,
};

use crossbeam::atomic::AtomicCell;
use rand::prelude::SliceRandom;

#[cfg(feature = "gui")]
use gtk::{
    glib::{self, clone},
    prelude::*,
    Application, ApplicationWindow, Button,
};
use tokio::sync::mpsc;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
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
    println!("Solving using bfs...");
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
    println!("Solving using dfs...");
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
    println!("Solving using a-star...");

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

async fn solve_astar_live(
    board: [[u8; 3]; 3],
    channel: mpsc::UnboundedSender<(State, Vec<Move>, u32)>,
) {
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
        let _ = channel
            .send((state.clone(), moves.clone(), nodes_visited))
            .unwrap();
        if state.is_goal() {
            eprintln!("Nodes visited: {}", nodes_visited);
            return;
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

fn is_solvable_flat(board: [u8; 9]) -> bool {
    let mut inversions = 0;
    for i in 0..8 {
        for j in i + 1..9 {
            if board[i] != 0 && board[j] != 0 && board[i] > board[j] {
                inversions += 1;
            }
        }
    }
    inversions % 2 == 0
}

#[cfg(feature = "gui")]
#[tokio::main]
async fn main() -> glib::ExitCode {
    let app = Application::builder()
        .application_id("me.vypal.puzzle8")
        .build();

    app.connect_activate(|app| {
        let window = ApplicationWindow::builder()
            .application(app)
            .title("Puzzle 8 Solver")
            .default_width(350)
            .default_height(70)
            .build();

        let scroll_area = gtk::ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Automatic)
            .vscrollbar_policy(gtk::PolicyType::Automatic)
            .build();

        let grid = gtk::Grid::builder()
            .row_spacing(5)
            .column_spacing(5)
            .build();

        let vbox = gtk::Box::builder()
            .orientation(gtk::Orientation::Vertical)
            .spacing(5)
            .build();

        let (tx, mut rx) = mpsc::unbounded_channel::<(State, Vec<Move>, u32)>();

        let board = Arc::new(AtomicCell::new([[1, 2, 3], [4, 5, 6], [7, 8, 0]]));

        let _ = tx
            .send((State::new(board.load()).clone(), Vec::new(), 0))
            .unwrap();

        glib::spawn_future_local(clone!(
            #[weak]
            grid,
            #[weak]
            window,
            async move {
                while !rx.is_closed() {
                    let mut buf = Vec::with_capacity(100);
                    rx.recv_many(&mut buf, 100).await;
                    let (state, moves, nodes_visited) = buf.into_iter().last().unwrap();

                    let label = gtk::Label::new(Some(&format!("Nodes visited: {}", nodes_visited)));
                    grid.remove_column(3);
                    grid.remove_row(3);
                    grid.remove_row(2);
                    grid.remove_row(1);
                    grid.remove_row(0);

                    grid.attach(&label, 0, 0, 1, 1);
                    for (i, row) in state.board.iter().enumerate() {
                        for (j, cell) in row.iter().enumerate() {
                            let button = Button::builder().label(&cell.to_string()).build();
                            grid.attach(&button, j as i32, i as i32 + 1, 1, 1);
                        }
                    }
                    for (i, mv) in moves.iter().enumerate() {
                        let label = gtk::Label::new(Some(&format!("{}) {:?}", i + 1, mv)));
                        grid.attach(&label, 3, i as i32 + 1, 1, 1);
                    }

                    window.present();
                }
            }
        ));

        let run_button = Button::builder().label("Run").hexpand(true).build();

        let tx_clone = tx.clone();
        let board_clone = board.clone();
        run_button.connect_clicked(clone!(
            #[weak]
            grid,
            #[weak]
            window,
            move |_| {
                let tx = tx_clone.clone();
                let board = board_clone.clone();
                if is_solvable(board.load()) {
                    let tx = tx.clone();
                    let value = board.load();
                    tokio::spawn(async move {
                        solve_astar_live(value, tx).await;
                    });
                } else {
                    let label = gtk::Label::new(Some("Unsolvable board"));
                    grid.attach(&label, 0, 0, 1, 1);
                    window.present();
                }
            }
        ));

        let randomzie_button = Button::builder().label("Randomize").hexpand(true).build();

        randomzie_button.connect_clicked(move |_| {
            let b = board.load();
            let mut b = [
                b[0][0], b[0][1], b[0][2], b[1][0], b[1][1], b[1][2], b[2][0], b[2][1], b[2][2],
            ];

            b.shuffle(&mut rand::rng());
            while !is_solvable_flat(b) {
                b.shuffle(&mut rand::rng());
            }
            board.store([[b[0], b[1], b[2]], [b[3], b[4], b[5]], [b[6], b[7], b[8]]]);
            let _ = tx
                .send((State::new(board.load()).clone(), Vec::new(), 0))
                .unwrap();
        });

        vbox.append(&run_button);
        vbox.append(&randomzie_button);
        vbox.append(&grid);
        scroll_area.set_child(Some(&vbox));
        window.set_child(Some(&scroll_area));

        window.present();
    });

    app.run()
}

#[cfg(feature = "tui")]
fn main() {
    use std::time::Instant;

    let board = Arc::new(AtomicCell::new([[1, 2, 3], [4, 5, 6], [7, 8, 0]]));

    println!("Initial board: {:?}", board.load());

    let b = board.load();
    let mut b = [
        b[0][0], b[0][1], b[0][2], b[1][0], b[1][1], b[1][2], b[2][0], b[2][1], b[2][2],
    ];

    b.shuffle(&mut rand::rng());
    while !is_solvable_flat(b) {
        b.shuffle(&mut rand::rng());
    }
    board.store([[b[0], b[1], b[2]], [b[3], b[4], b[5]], [b[6], b[7], b[8]]]);

    println!("After shuffle: {:?}", board.load());

    let state = State::new(board.load());
    println!("Initial state: {:?}", state);

    let start = Instant::now();

    #[cfg(feature = "astar")]
    let (state, moves, nodes_visited) = solve_astar(board.load());

    #[cfg(feature = "bfs")]
    let (state, moves, nodes_visited) = solve_bfs(board.load());

    #[cfg(feature = "dfs")]
    let (state, moves, nodes_visited) = solve_dfs(board.load());

    let end = start.elapsed();

    println!("Took {:?}", end);
    println!("Final state: {:?}", state);
    println!("Moves made: {:?}", moves);
    println!("Nodes visited: {:?}", nodes_visited);
}
