

### <h1 align="center" id="title"> Documentation of update_plot </h1>


Help on method update_plot in module igm:

update_plot(force=False) method of igm.Igm instance
    Plot thickness, velocity, mass balance



### <h1 align="center" id="title"> Parameters of update_plot </h1>


``` 

usage: [-h] [--varplot VARPLOT] [--varplot_max VARPLOT_MAX]

optional arguments:
  -h, --help            show this help message and exit
  --varplot VARPLOT     variable to plot
  --varplot_max VARPLOT_MAX
                        maximum value of the varplot variable used to adjust
                        the scaling of the colorbar
``` 



### <h1 align="center" id="title"> Code of update_plot </h1>


```python 

    def update_plot(self, force=False):
        """
        Plot thickness, velocity, mass balance
        """

        if force | (self.saveresult & self.config.plot_result):

            firstime = False
            if not hasattr(self, "already_called_update_plot"):
                self.already_called_update_plot = True
                self.tcomp["Outputs plot"] = []
                firstime = True

            self.tcomp["Outputs plot"].append(time.time())

            if self.config.varplot == "velbar_mag":
                self.velbar_mag = self.getmag(self.ubar, self.vbar)

            if firstime:

                self.fig = plt.figure(dpi=200)
                self.ax = self.fig.add_subplot(1, 1, 1)
                self.ax.axis("off")
                im = self.ax.imshow(
                    vars(self)[self.config.varplot],
                    origin="lower",
                    cmap="viridis",
                    vmin=0,
                    vmax=self.config.varplot_max,
                )
                if self.config.tracking_particles:
                    r = 1
                    self.ip = self.ax.scatter(
                        x=(self.xpos[::r] - self.x[0]) / self.dx,
                        y=(self.ypos[::r] - self.y[0]) / self.dx,
                        c=1 - self.rhpos[::r].numpy(),
                        vmin=0,
                        vmax=1,
                        s=0.5,
                        cmap="RdBu",
                    )
                self.ax.set_title("YEAR : " + str(self.t.numpy()), size=15)
                self.cbar = plt.colorbar(im)

            else:
                im = self.ax.imshow(
                    vars(self)[self.config.varplot],
                    origin="lower",
                    cmap="viridis",
                    vmin=0,
                    vmax=self.config.varplot_max,
                )
                if self.config.tracking_particles:
                    self.ip.set_visible(False)
                    r = 1
                    self.ip = self.ax.scatter(
                        x=(self.xpos[::r] - self.x[0]) / self.dx,
                        y=(self.ypos[::r] - self.y[0]) / self.dx,
                        c=1 - self.rhpos[::r].numpy(),
                        vmin=0,
                        vmax=1,
                        s=0.5,
                        cmap="RdBu",
                    )
                self.ax.set_title("YEAR : " + str(self.t.numpy()), size=15)

            if self.config.plot_live:
                clear_output(wait=True)
                display(self.fig)

            else:
                plt.savefig(
                    os.path.join(
                        self.config.working_dir,
                        self.config.varplot
                        + "-"
                        + str(self.t.numpy()).zfill(4)
                        + ".png",
                    ),
                    bbox_inches="tight",
                    pad_inches=0.2,
                )

            self.tcomp["Outputs plot"][-1] -= time.time()
            self.tcomp["Outputs plot"][-1] *= -1

``` 

