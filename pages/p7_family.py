"""pages/p7_family.py — Family Overview page."""
from __future__ import annotations
import streamlit as st
from pages.p5_accounts import render as accounts_render

def render(**kwargs):
    accounts_render(**kwargs)
