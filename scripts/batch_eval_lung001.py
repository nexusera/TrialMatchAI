#!/usr/bin/env python3
"""Batch-evaluate patient-lung-001 with one GT source across many predictions."""

from batch_eval_common import main_with_preset

if __name__ == "__main__":
    raise SystemExit(main_with_preset("lung001"))
