import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolBase

import numpy as np

matplotlib.rcParams["toolbar"] = "toolmanager"


class NewTool(ToolBase):
    image = r"kinect2.png"


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1, 2, 3], label="legend")
ax.legend()
tm = fig.canvas.manager.toolmanager
tm.add_tool("newtool", NewTool)
fig.canvas.manager.toolbar.add_tool(tm.get_tool("newtool"), "toolgroup")
plt.show()