#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path
import torch

from arch_builder.audio_frontend import AudioMatrixClassifier, build_frontend
from tools.project_probe.export_program_report import generate_markdown_report

def test_program_export():
    print("Testing Program Export Report generator...")
    
    # Build a small dummy model to test generation logic
    dim = 8
    slots = 2
    frontend = build_frontend("raw", slots=slots, dim=dim)
    model = AudioMatrixClassifier(
        frontend=frontend,
        classes=2,
        dim=dim,
        slots=slots,
        layers=2,
        top_k=4,
        sim_rank=4,
        enable_vnext=True,
        enable_scanner_feedback_memory=True,
    )
    
    # Initialize some mock feedback data
    pm = model.backbone.pm
    pm.feedback_gain_ema.fill_(0.1)
    pm.feedback_regret_ema.fill_(0.01)
    pm.feedback_count.fill_(1)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "test_report.md"
        report = generate_markdown_report(model, str(report_path))
        
        # Verify file creation and basic content
        assert report_path.exists(), "Report file was not created"
        content = report_path.read_text()
        
        print("Generated report snippet:")
        print(content[:500])
        
        assert "# Synthesized Neural Program Report" in content
        assert "Global Primitive Usage Score" in content
        assert "Program Blueprint" in content
        assert "Program Execution Parameters" in content
        
    print("PROGRAM_EXPORT_PASS")

if __name__ == "__main__":
    test_program_export()
