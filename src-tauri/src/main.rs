// src-tauri/src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tibert_classical_lib::run;

fn main() {
    run();
}
