//! Document loading: markdown, plain text, and PDF.
//!
//! Everything is local. PDF text comes from the pure-Rust `pdf-extract`
//! crate; no external tools, no network.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// A loaded document: raw text plus its origin.
#[derive(Debug)]
pub struct Document {
    /// Absolute (or as-given) file path.
    pub path: PathBuf,
    /// Extracted text.
    pub text: String,
    /// Lowercase file extension, e.g. `"md"`, `"txt"`, `"pdf"`.
    pub kind: String,
}

/// Load one file. Returns an `Unsupported` error for unknown extensions and
/// an `InvalidData` error when PDF extraction fails.
///
/// Markdown code fences are stripped: code blocks fragment into
/// punctuation-heavy shards that pollute extractive answers.
pub fn load_document(path: &Path) -> io::Result<Document> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();
    let text = match ext.as_str() {
        "md" | "markdown" => {
            let raw = String::from_utf8_lossy(&fs::read(path)?).into_owned();
            strip_code_fences(&raw)
        }
        "txt" | "text" => String::from_utf8_lossy(&fs::read(path)?).into_owned(),
        "pdf" => pdf_extract::extract_text(path).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("pdf extract failed: {e}"),
            )
        })?,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("unsupported file type: .{other} ({})", path.display()),
            ))
        }
    };
    Ok(Document {
        path: path.to_path_buf(),
        text,
        kind: ext,
    })
}

/// Recursively walk `dir` and load every supported document, sorted by path
/// for deterministic ingestion order. Hidden files and directories (names
/// starting with `.`) are skipped.
pub fn ingest_dir(dir: &Path) -> io::Result<Vec<Document>> {
    let mut files = Vec::new();
    collect_files(dir, &mut files)?;
    files.sort();
    let mut docs = Vec::new();
    for f in files {
        match load_document(&f) {
            Ok(doc) => docs.push(doc),
            Err(e) if e.kind() == io::ErrorKind::Unsupported => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(docs)
}

/// Remove fenced code blocks (``` ... ```) from markdown text.
/// Unclosed fences strip to end of input.
fn strip_code_fences(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_fence = false;
    for line in text.split_inclusive('\n') {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if !in_fence {
            out.push_str(line);
        }
    }
    out
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            collect_files(&path, out)?;
        } else if ft.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write(path: &Path, content: &[u8]) {
        let mut f = fs::File::create(path).unwrap();
        f.write_all(content).unwrap();
    }

    #[test]
    fn loads_markdown_and_text() {
        let dir = tempfile::tempdir().unwrap();
        let md = dir.path().join("a.md");
        let txt = dir.path().join("b.txt");
        write(&md, b"# Title\n\nSome content here.");
        write(&txt, "plain text".as_bytes());
        let docs = ingest_dir(dir.path()).unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs.iter().any(|d| d.kind == "md"));
        assert!(docs.iter().any(|d| d.text.contains("plain text")));
    }

    #[test]
    fn skips_unsupported_and_hidden() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir.path().join("a.bin"), b"\x00\x01\x02");
        write(&dir.path().join(".hidden.md"), b"hidden");
        write(&dir.path().join("ok.md"), b"visible");
        let docs = ingest_dir(dir.path()).unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].text.contains("visible"));
    }

    #[test]
    fn unsupported_extension_errors() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.xyz");
        write(&p, b"data");
        let err = load_document(&p).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
    }

    #[test]
    fn extracts_text_from_minimal_pdf() {
        // Hand-built minimal PDF 1.4 with one text-showing content stream.
        let pdf = minimal_pdf("Hello PDF world");
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("doc.pdf");
        write(&p, &pdf);
        let doc = load_document(&p).unwrap();
        assert_eq!(doc.kind, "pdf");
        assert!(
            doc.text.contains("Hello PDF world"),
            "extracted text was: {:?}",
            doc.text
        );
    }

    #[test]
    fn corrupt_pdf_is_an_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.pdf");
        write(&p, b"%PDF-1.4\nthis is not a real pdf");
        let err = load_document(&p).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn code_fences_stripped_from_markdown() {
        let md = "# Title\n\n```rust\nlet x = 1;\n```\n\nReal prose here.\n";
        let stripped = strip_code_fences(md);
        assert!(!stripped.contains("let x = 1;"));
        assert!(!stripped.contains("```"));
        assert!(stripped.contains("Real prose here."));
    }

    #[test]
    fn unclosed_fence_strips_to_end() {
        let md = "Prose.\n```\ncode code code";
        assert_eq!(strip_code_fences(md), "Prose.\n");
    }

    /// Build the smallest valid PDF that `pdf-extract` (via lopdf) can parse.
    fn minimal_pdf(text: &str) -> Vec<u8> {
        // Escape parentheses in the text for the PDF string literal.
        let escaped = text
            .replace('\\', "\\\\")
            .replace('(', "\\(")
            .replace(')', "\\)");
        let stream = format!("BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET");
        let mut pdf = Vec::new();
        let mut offsets = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.4\n");
        offsets.push(pdf.len());
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        offsets.push(pdf.len());
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
        offsets.push(pdf.len());
        pdf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n");
        offsets.push(pdf.len());
        pdf.extend_from_slice(
            format!(
                "4 0 obj\n<< /Length {} >>\nstream\n{}\nendstream\nendobj\n",
                stream.len(),
                stream
            )
            .as_bytes(),
        );
        offsets.push(pdf.len());
        pdf.extend_from_slice(
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        );
        let xref_pos = pdf.len();
        pdf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
        for off in &offsets {
            pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n")
                .as_bytes(),
        );
        pdf
    }
}
