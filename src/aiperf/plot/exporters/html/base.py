# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HTML-specific base exporter functionality.

Provides HTML export methods including Jinja2 template rendering,
CSS/JavaScript bundling, and Plotly.js embedding.
"""

from pathlib import Path

import jinja2

from aiperf.plot.constants import (
    DARK_THEME_COLORS,
    LIGHT_THEME_COLORS,
    NVIDIA_BORDER_DARK,
    NVIDIA_BORDER_LIGHT,
    NVIDIA_CARD_BG,
    NVIDIA_DARK,
    NVIDIA_DARK_BG,
    NVIDIA_GRAY,
    NVIDIA_GREEN,
    NVIDIA_TEXT_LIGHT,
    NVIDIA_WHITE,
    PLOT_FONT_FAMILY,
    PlotTheme,
)
from aiperf.plot.dashboard.styling import get_all_themes_css
from aiperf.plot.exporters.base import BaseExporter

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


class BaseHTMLExporter(BaseExporter):
    """
    Base class for HTML export functionality.

    Provides HTML-specific methods like Jinja2 template rendering,
    CSS/JavaScript bundling, and HTML file writing.
    """

    def __init__(
        self,
        output_dir: Path,
        theme: PlotTheme = PlotTheme.LIGHT,
        color_pool_size: int = 10,
    ) -> None:
        """
        Initialize the HTML exporter.

        Args:
            output_dir: Directory where exported files will be saved
            theme: Theme to use for plots (LIGHT or DARK)
            color_pool_size: Number of colors for group assignments
        """
        super().__init__(output_dir, theme, color_pool_size)
        self._jinja_env: jinja2.Environment | None = None

    def _get_jinja_env(self) -> jinja2.Environment:
        """
        Get configured Jinja2 environment with template loader.

        Returns:
            Jinja2 Environment configured for HTML templates
        """
        if self._jinja_env is None:
            self._jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
                autoescape=jinja2.select_autoescape(["html", "xml"]),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        return self._jinja_env

    def _render_template(self, template_name: str, context: dict) -> str:
        """
        Render a Jinja2 template with the given context.

        Args:
            template_name: Name of the template file (e.g., 'base.html.j2')
            context: Dictionary of template variables

        Returns:
            Rendered HTML string
        """
        env = self._get_jinja_env()
        template = env.get_template(template_name)
        return template.render(**context)

    def _get_css_bundle(self) -> str:
        """
        Get CSS bundle combining dashboard styling and HTML-specific styles.

        Reuses the dashboard styling for visual consistency, then adds
        HTML-specific overrides and additions.

        Returns:
            CSS string with NVIDIA theming for both light and dark modes
        """
        dashboard_css = get_all_themes_css()

        css_file = STATIC_DIR / "styles.css"
        html_specific_css = ""
        if css_file.exists():
            html_specific_css = css_file.read_text(encoding="utf-8")
        else:
            html_specific_css = self._generate_css()

        return (
            dashboard_css
            + "\n\n/* HTML Export Specific Styles */\n"
            + html_specific_css
        )

    def _generate_css(self) -> str:
        """
        Generate CSS with NVIDIA theming.

        Returns:
            CSS string with theme variables and component styles
        """
        return f"""
:root {{
    --color-primary: {NVIDIA_GREEN};
    --color-background: {NVIDIA_WHITE};
    --color-paper: {NVIDIA_WHITE};
    --color-text: {NVIDIA_DARK};
    --color-text-secondary: {NVIDIA_GRAY};
    --color-border: {NVIDIA_BORDER_LIGHT};
    --color-grid: {NVIDIA_BORDER_LIGHT};
    --font-family: {PLOT_FONT_FAMILY};
}}

.theme-dark {{
    --color-background: {NVIDIA_DARK_BG};
    --color-paper: {NVIDIA_CARD_BG};
    --color-text: {NVIDIA_TEXT_LIGHT};
    --color-text-secondary: {NVIDIA_GRAY};
    --color-border: {NVIDIA_BORDER_DARK};
    --color-grid: {NVIDIA_BORDER_DARK};
}}

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: var(--font-family);
    background-color: var(--color-background);
    color: var(--color-text);
    min-height: 100vh;
}}

.header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 24px;
    background-color: var(--color-paper);
    border-bottom: 1px solid var(--color-border);
    position: sticky;
    top: 0;
    z-index: 100;
}}

.header-logo {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

.nvidia-logo {{
    font-weight: 700;
    font-size: 1.2rem;
    color: var(--color-primary);
}}

.aiperf-title {{
    font-size: 1rem;
    color: var(--color-text-secondary);
}}

.header-controls {{
    display: flex;
    gap: 8px;
}}

.theme-toggle-btn {{
    background: none;
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 6px 12px;
    cursor: pointer;
    font-size: 1rem;
}}

.theme-toggle-btn:hover {{
    background-color: var(--color-border);
}}

.main-container {{
    display: flex;
    min-height: calc(100vh - 57px);
}}

.sidebar {{
    width: 280px;
    background-color: var(--color-paper);
    border-right: 1px solid var(--color-border);
    padding: 16px;
    overflow-y: auto;
    flex-shrink: 0;
    transition: width 0.2s, padding 0.2s;
}}

.sidebar.collapsed {{
    width: 0;
    padding: 0;
    overflow: hidden;
}}

.sidebar-toggle {{
    position: fixed;
    left: 8px;
    top: 70px;
    z-index: 101;
    background: var(--color-paper);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 8px;
    cursor: pointer;
    font-size: 1.2rem;
}}

.sidebar-section {{
    margin-bottom: 24px;
}}

.section-header {{
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--color-text);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.control-label {{
    display: block;
    font-size: 0.85rem;
    color: var(--color-text-secondary);
    margin-bottom: 4px;
}}

.dropdown {{
    width: 100%;
    padding: 8px;
    border: 1px solid var(--color-border);
    border-radius: 4px;
    background-color: var(--color-paper);
    color: var(--color-text);
    font-size: 0.9rem;
    margin-bottom: 8px;
}}

.btn-primary {{
    width: 100%;
    padding: 10px 16px;
    background-color: var(--color-primary);
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 0.9rem;
    cursor: pointer;
    font-weight: 500;
}}

.btn-primary:hover {{
    opacity: 0.9;
}}

.btn-secondary {{
    width: 100%;
    padding: 10px 16px;
    background-color: transparent;
    color: var(--color-text);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    font-size: 0.9rem;
    cursor: pointer;
    margin-top: 8px;
}}

.btn-secondary:hover {{
    background-color: var(--color-border);
}}

.run-selector-container {{
    max-height: 300px;
    overflow-y: auto;
}}

.run-checkbox {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    font-size: 0.85rem;
    cursor: pointer;
}}

.run-checkbox input {{
    cursor: pointer;
}}

.plot-grid-container {{
    flex: 1;
    padding: 16px;
    overflow-y: auto;
}}

.plot-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
}}

.plot-container {{
    background-color: var(--color-paper);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    position: relative;
    min-height: 400px;
}}

.plot-container.size-half {{
    width: calc(50% - 8px);
}}

.plot-container.size-full {{
    width: 100%;
    min-height: 600px;
}}

@media (max-width: 1200px) {{
    .plot-container.size-half {{
        width: 100%;
    }}
}}

.plot-toolbar {{
    position: absolute;
    top: 8px;
    right: 8px;
    display: flex;
    gap: 4px;
    z-index: 10;
}}

.plot-toolbar button {{
    background: var(--color-paper);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 4px 8px;
    cursor: pointer;
    font-size: 0.9rem;
    opacity: 0.7;
}}

.plot-toolbar button:hover {{
    opacity: 1;
}}

.plot-area {{
    width: 100%;
    height: 100%;
    min-height: inherit;
}}

.plot-add-slot {{
    width: calc(50% - 8px);
    min-height: 200px;
    background-color: var(--color-paper);
    border: 2px dashed var(--color-border);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    opacity: 0.6;
    transition: opacity 0.2s;
}}

.plot-add-slot:hover {{
    opacity: 1;
}}

.add-icon {{
    font-size: 2rem;
    color: var(--color-primary);
}}

.add-text {{
    font-size: 0.9rem;
    color: var(--color-text-secondary);
    margin-top: 8px;
}}

.modal-overlay {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}}

.modal-overlay.active {{
    display: flex;
}}

.modal {{
    background-color: var(--color-paper);
    border-radius: 8px;
    padding: 24px;
    max-width: 500px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
}}

.modal-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}}

.modal-title {{
    font-size: 1.2rem;
    font-weight: 600;
}}

.modal-close {{
    background: none;
    border: none;
    font-size: 1.5rem;
    cursor: pointer;
    color: var(--color-text-secondary);
}}

.modal-body {{
    margin-bottom: 16px;
}}

.modal-footer {{
    display: flex;
    justify-content: flex-end;
    gap: 8px;
}}

.form-group {{
    margin-bottom: 16px;
}}

.form-group label {{
    display: block;
    margin-bottom: 4px;
    font-weight: 500;
}}

.checkbox-group {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}}

.hidden {{
    display: none !important;
}}
"""

    def _get_js_bundle(self) -> str:
        """
        Get JavaScript bundle, either from static file or generated.

        Returns:
            JavaScript string with all app logic
        """
        js_file = STATIC_DIR / "app.js"
        if js_file.exists():
            return js_file.read_text(encoding="utf-8")

        return self._generate_minimal_js()

    def _generate_minimal_js(self) -> str:
        """
        Generate minimal JavaScript as fallback.

        Returns:
            Basic JavaScript string
        """
        return """
console.error('app.js not found. Please ensure static/app.js exists.');
"""

    def _get_plotly_js(self) -> str:
        """
        Get Plotly.js script tag for embedding.

        Returns a CDN script tag for Plotly.js. Note that this requires
        internet access to load the library.

        Returns:
            Script tag linking to Plotly.js CDN
        """
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'

    def _get_theme_colors(self) -> dict:
        """
        Get theme colors based on current theme setting.

        Returns:
            Dict of color values for the current theme
        """
        if self.theme == PlotTheme.DARK:
            return DARK_THEME_COLORS
        return LIGHT_THEME_COLORS

    def _write_html_file(self, content: str, filename: str) -> Path:
        """
        Write HTML content to file.

        Args:
            content: HTML string to write
            filename: Output filename

        Returns:
            Path to the written file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / filename
        output_path.write_text(content, encoding="utf-8")
        self.info(f"HTML export saved to {output_path}")
        return output_path
