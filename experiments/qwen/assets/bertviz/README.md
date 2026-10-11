# Local BertViz browser dependencies

BertViz 1.4.1 loads these libraries from cdnjs by default. The playground embeds
the same versions in an isolated iframe so the visualization works offline.
The JavaScript library copyright/license comments remain in the files.

- D3 5.7.0: https://cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min.js (BSD-3-Clause)
- jQuery 2.0.0: https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.0/jquery.min.js (MIT)
- BertViz: https://github.com/jessevig/bertviz (Apache-2.0)

These versions match BertViz's own requirements. The iframe permits scripts,
without same-origin access to Jupyter. Do not use these assets as a general
website dependency bundle.
