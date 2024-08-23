A browser-based dashboard for performing visual analysis as a part of the exploratory modeling of complex models. Currently supports analysis of any model so long as the model run is executed with the EMA Workbench [1]. At present, can only visualize time-series data with no spatial dimension. 

Principally for seeing the shape of a large set of time-series, especially highlighting runs at any region of a model's uncertain parameter space you care to study. Clustered outcomes can also be grouped and visualized, or runs can be coloured sequentially according to one input parameter.

Implemented using Plotly's Dash framework.

If you're working with the EMA Workbench in your CAS/SES/CHANS/etc. study, feel free to use this and adapt it. As Dash open source can only be used locally, please do not host a dashboard built with this code on the Internet. Reach out to [@ghsher](https://x.com/ghsher) on Twitter/X if you have any questions (or contact me via the info in my GitHub profile).

---

[1] Kwakkel, J. H. (2013). Exploratory Modeling and Analysis Workbench. https://github.com/quaquel/EMAworkbench
