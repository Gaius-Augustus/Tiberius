"""Tests for the command line switches of the spliced-stop handling."""

import sys
from unittest import mock

from tiberius.tiberius_args import parseCmd


def _parse(*flags):
    argv = ["tiberius.py", "--genome", "g.fa", "--model_cfg", "m", *flags]
    with mock.patch.object(sys, "argv", argv):
        return parseCmd()


def test_spliced_stop_defaults_keep_current_behavior():
    args = _parse()
    assert args.no_spliced_stop_reprediction is False
    assert args.keep_inframe_stops is False


def test_no_spliced_stop_reprediction_flag():
    args = _parse("--no_spliced_stop_reprediction")
    assert args.no_spliced_stop_reprediction is True
    assert args.keep_inframe_stops is False


def test_keep_inframe_stops_flag():
    args = _parse("--keep_inframe_stops")
    assert args.keep_inframe_stops is True
    assert args.no_spliced_stop_reprediction is False
