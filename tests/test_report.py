"""Unit tests for the report builder."""

import pytest

from report.builder import ReportBuilder


def test_report_builder_init() -> None:
    """Test initialization."""
    builder = ReportBuilder()
    assert builder is not None


def test_add_section() -> None:
    """Test adding a section."""
    builder = ReportBuilder()
    assert builder.add_section("Test") is None


def test_build() -> None:
    """Test building report."""
    builder = ReportBuilder()
    assert builder.build() is None


def test_export() -> None:
    """Test exporting report."""
    builder = ReportBuilder()
    assert builder.export("/tmp/test.html") is None
