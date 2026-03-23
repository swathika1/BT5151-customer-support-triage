"""Compatibility helpers for running Gradio demos from notebooks.

This project currently uses ``gradio==4.26.0``. In newer environments,
``starlette.templating.Jinja2Templates.TemplateResponse`` expects the request
object as the first positional argument, while older Gradio versions still call
it with ``(template_name, context)``.

The helper below patches Gradio's shared template object at runtime so notebook
prototypes can keep working without editing site-packages.
"""

from __future__ import annotations

import inspect
from typing import Any

import gradio as gr
from gradio import routes
from starlette.templating import Jinja2Templates


def apply_template_response_compat_patch() -> bool:
    """Patch Gradio's template renderer when Starlette uses the new signature.

    Returns ``True`` when the patch is active, otherwise ``False``.
    """

    params = list(inspect.signature(Jinja2Templates.TemplateResponse).parameters)
    needs_patch = params[:3] == ["self", "request", "name"]
    if not needs_patch:
        return False

    current = routes.templates.TemplateResponse
    if getattr(current, "__name__", "") == "_template_response_compat":
        return True

    def _template_response_compat(
        name: str,
        context: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ):
        request = context["request"]
        return current(request, name, context, *args, **kwargs)

    routes.templates.TemplateResponse = _template_response_compat
    return True


def launch_notebook_demo(
    demo: gr.Blocks,
    *,
    server_name: str = "127.0.0.1",
    server_port: int | None = None,
    share: bool = False,
    debug: bool = True,
):
    """Launch a Gradio demo from a notebook in a browser tab.

    ``inline=False`` and ``_frontend=False`` avoid VS Code/Jupyter localhost
    embedding issues. ``inbrowser=True`` opens the local URL directly.
    """

    apply_template_response_compat_patch()

    try:
        demo.close()
    except Exception:
        pass

    return demo.queue().launch(
        share=share,
        debug=debug,
        prevent_thread_lock=True,
        server_name=server_name,
        server_port=server_port,
        inline=False,
        inbrowser=True,
        _frontend=False,
    )
