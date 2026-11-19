pub mod dsp;
pub mod media;
pub mod model;
pub mod spectro;
pub mod sync_algo;
pub mod timeline;

use anyhow::Result;
use sha1::{Digest, Sha1};
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

pub fn file_key(p: &Path) -> Result<String> {
    let meta = fs::metadata(p)?;
    let mtime_ns = (meta.mtime() as i128) * 1_000_000_000 + (meta.mtime_nsec() as i128);
    let mut hasher = Sha1::new();
    hasher.update(format!("{}|{}|{mtime_ns}", p.display(), meta.len()).as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    Ok(hex[..20].to_string())
}
