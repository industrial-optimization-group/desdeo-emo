def plot_init_(self):
        """Initialize plot objects."""
        obj = self.objectives
        num_obj = obj.shape[1]
        if num_obj == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=Axes3D.name)
        else:
            fig, ax = plt.subplots()
        plt.ion()
        plt.show()
        self.plot_objectives(fig, ax)
        return (fig, ax)

    def plot_objectives(self, fig, ax):
        """Plot the objective values of individuals in notebook. This is a hack."""
        obj = self.objectives
        ref = self.reference_vectors.values
        num_samples, num_obj = obj.shape
        ax.clear()
        if num_obj == 2:
            plt.scatter(obj[:, 0], obj[:, 1])
        elif num_obj == 3:
            # ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2])
            # ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2])
            trace = go.Scatter3d(x=obj[:, 0], y=obj[:, 1], z=obj[:, 2], mode="markers")
            figure = go.Figure(data=[trace])
            py.offline.plot(figure, filename="3dscatter")
        else:
            objectives = pd.DataFrame(obj)
            # objectives["why"] = objectives[0]
            # color = plt.cm.rainbow(np.linspace(0, 1, len(objectives.index)))
            # ax.clear()
            # ax = parallel_coordinates(objectives, "why", ax=ax, color=color)
            # ax.get_legend().remove()
            data = go.Parcoords(
                line=dict(
                    color=objectives[0],
                    colorscale="Viridis",
                    showscale=True,
                    cmin=min(objectives[0]),
                    cmax=max(objectives[0]),
                ),
                dimensions=list(
                    [
                        dict(
                            range=[min(objectives[0]), max(objectives[0])],
                            label="f0",
                            values=objectives[0],
                        ),
                        dict(
                            range=[min(objectives[1]), max(objectives[1])],
                            label="f1",
                            values=objectives[1],
                        ),
                        dict(
                            range=[min(objectives[2]), max(objectives[2])],
                            label="f2",
                            values=objectives[2],
                        ),
                        dict(
                            range=[min(objectives[3]), max(objectives[3])],
                            label="f3",
                            values=objectives[3],
                        ),
                    ]
                ),
            )
            py.offline.plot([data], filename="parallelplot.html")
        # fig.canvas.draw()
