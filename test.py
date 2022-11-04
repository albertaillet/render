# %%
import numpy as np
import plotly.express as px

size = 100
r = 10
x0, y0 = 5, 0

# %%
range = np.arange(-size, size)
x, y = np.meshgrid(range, range)
z0 = (r ** 2 - x0 ** 2 - y0 ** 2)
z0 = np.sqrt(z0)

# %%
z = r**2 - x**2 - y**2
z = np.sqrt(z, where=(z > 0))
q = (x ** 2 + y ** 2) ** 0.5
b = (x * x0 + y * y0 + z * z0) / r ** 2
fig = px.imshow(b)
fig.update_layout(
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    coloraxis_showscale=False,
)
# %%
