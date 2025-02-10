"""
Plotting template for a bowling pitch map with overlaid 2D density, which utilises the front view of the plot_wicket_3d
function.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.font_manager as fm
import mpl_toolkits.mplot3d.art3d as art3d
from pitch_view.wicket_3d import plot_wicket_3d
from pitch_view.plotting_utils import add_title_axis


# Helper function to get gaussian kde for plotting - returns X,Y grid and Z density
def get_density(data):
    xmin = min(data[:, 0])
    xmax = max(data[:, 0])
    ymin = min(data[:, 1])
    ymax = max(data[:, 1])

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # X, Y = np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100)
    # X, Y = np.meshgrid(X, Y)
    positions = np.vstack([X.ravel(), Y.ravel()])

    kernel = gaussian_kde(data.T, bw_method=0.30)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z

def pitch_densitymap(xy, title='', subtitle_1='', subtitle_2=''):

    """ Plots a heatmap overlaid on wicket_3d front view, using a specified values array for square shading

    ----------
    xy: A 2d array
        The x and y coordinates of the delivery pitching locations
    title: A string
        The plot title
    subtitle_1: A string
        The plot's 1st subtitle
    subtitle_2: A string
        The plot's 2nd subtitle

    Returns
    -------
    matplotlib.axes.Axes"""

    # Define some styling
    # pitch_colour = 'white'
    # wicket_colour = '#f5f6fa'
    # marking_colour = 'cornflowerblue'
    # stump_colour = 'slategray'
    # outline_colour = 'lightsteelblue'
    # title_colour = '#080a2e'
    # subtitle_colour = '#9e9fa3'
    # fname = 'fonts/AlumniSans-SemiBold.ttf'
    # fp = fm.FontProperties(fname=fname)

    pitch_colour = 'white'
    wicket_colour = '#f5f6fa'
    marking_colour = '#595959'
    stump_colour = 'slategray'
    outline_colour = '#595959'
    title_colour = '#080a2e'
    subtitle_colour = '#9e9fa3'
    fname = './pitch_view/AlumniSans-SemiBold.ttf'
    fp = fm.FontProperties(fname=fname)

    X, Y, Z = get_density(xy)

    # df = new_df_bowler.copy()
    # boundaries_df = df[(df['IsFour'] == 1) | (df['IsSix'] == 1)]
    # wickets_df = df[(df['IsWicket'] == 1)]
    # dots_df = df[(df['IsDotball'] == 1)]
    # runs_df = df[pd.to_numeric(df['BallRuns'], errors='coerce').fillna(0).astype(int).gt(0) & (df['IsFour'] == 0) & (df['IsSix'] == 0)]
    # Plot a 2D KDE plot of the delivery pitch locations on a 3D pitch

    fig = plt.figure()
    fig.set_size_inches(8, 5)
    fig.subplots_adjust(left=0,
                        right=1,
                        bottom=-0.2,
                        top=1)  # Get rid of some excess whitespace - adjust to taste

    # ax = plt.gca(projection='3d')  # We'll plot on a 3D axis
    ax = fig.add_subplot(111, projection='3d')
    # We have data for a 3D surface plot, but we want to plot a 2D surface on the xy plane, so we'll set the Z axis
    # to zeros
    z_axis = np.zeros(X.shape)

    # We will manually set the colours of the surface based on the actual Z data using facecolors argument of
    # ax.plot_surface
    colours = plt.cm.PuRd(Z)

    # A trick to apply gradual alpha shading to the surface plot for cleaner looking visual

    for i in range(len(colours)):
        plane = colours[i]
        for j in range(len(plane)):
            row = plane[j]
            if (row[0:3].mean() >= 0.87) & (row[0:3].mean() < 0.92):
                row[3] = 0.25
            elif row[0:3].mean() >= 0.92:
                row[3] = 0

    # Plot the surfaces
    ax.plot_surface(X,
                    Y,
                    z_axis,
                    cmap='Purples',
                    facecolors=colours,
                    linewidth=1,
                    antialiased=False)

    # ax.scatter(-dots_df['bounce_y'], dots_df['bounce_x'],  
    #         c='green', 
    #         edgecolor='#FF8C00',  # Dark orange hex code
    #         alpha=0.7, 
    #         zorder=9, 
    #         s=30, 
    #         linewidth=1.2, 
    #         marker='o',
    #         label="Dots")
    
    # ax.scatter(-runs_df['bounce_y'], runs_df['bounce_x'], 
    #         c='yellow', 
    #         edgecolor='#FF8C00',  # Dark orange hex code
    #         alpha=0.7, 
    #         zorder=9, 
    #         s=30, 
    #         linewidth=1.2, 
    #         marker='o',
    #         label="Runs")
    
    # ax.scatter(-boundaries_df['bounce_y'], boundaries_df['bounce_x'],  
    #         c='red', 
    #         edgecolor='#FF8C00',  # Dark orange hex code
    #         alpha=1.0, 
    #         zorder=9, 
    #         s=30, 
    #         linewidth=1.2, 
    #         marker='o',
    #         label="4s/6s")
    
    # ax.scatter(-wickets_df['bounce_y'], wickets_df['bounce_x'], 
    #         c='blue', 
    #         edgecolor='#FF8C00',  # Dark orange hex code
    #         alpha=1.0, 
    #         zorder=9, 
    #         s=30, 
    #         linewidth=1.2, 
    #         marker='o',
    #         label="Wickets")
    # ax.legend()
    # Plot the pitch points as a scatter overlaid
    ax.scatter(xy[:, 0], xy[:, 1], c='blue', edgecolor='black', alpha=0.5, zorder=9)


    # Add titles and subtitles
    add_title_axis(fig,
                   title,
                   subtitle_1,
                   subtitle_2,
                   fp=fp,
                   title_colour=title_colour,
                   subtitle_colour=subtitle_colour)

    # Generate a cricket pitch on the axis we created
    plot_wicket_3d(ax,
                   view='front',
                   pitch_colour=pitch_colour,
                   marking_colour=marking_colour,
                   outline_colour=outline_colour,
                   stump_colour=stump_colour,
                   wicket_colour=wicket_colour)
    
    return fig
    # plt.show()

def bowling_average_heatmap(xy,
                            runs,
                            dismissals,
                            wickets,
                            title='Bowling Average Heatmap',
                            subtitle_1='',
                            subtitle_2='',
                            legend_title='Average',
                            min_balls=0,
                            cmap='Purples'):
    """
    Plots a heatmap showing batting average over a cricket pitch.

    ----------
    xy: A 2d array
        The x and y coordinates of the delivery pitching locations
    runs: A 1d array
        The number of runs scored for each ball
    dismissals: A 1d array
        Binary values indicating dismissals (1 for out, 0 for not out)
    title: A string
        The plot title
    subtitle_1: A string
        The plot's 1st subtitle
    subtitle_2: A string
        The plot's 2nd subtitle
    legend_title: A string
        The title of the value legend
    min_balls: An integer
        The minimum number of balls used for each heatmap zone
    cmap: Any valid matplotlib named colormap string
        The colour map used for the heatmap shading

    Returns
    -------
    matplotlib.axes.Axes
    """

    # Define some styling
    pitch_colour = 'white'
    wicket_colour = '#f5f6fa'
    marking_colour = '#595959'
    stump_colour = 'slategray'
    outline_colour = '#595959'
    title_colour = '#080a2e'
    subtitle_colour = '#9e9fa3'
    fname = './AlumniSans-SemiBold.ttf'
    fp = fm.FontProperties(fname=fname)

    # Define pitch bin edges
    XMIN, XMAX, XBIN = -1.83, 1.84, 0.2
    YMIN, YMAX, YBIN = 1, 15, 1
    x_edges = np.arange(XMIN, XMAX, XBIN)
    y_edges = np.arange(YMIN, YMAX, YBIN)

    # Combine data into a DataFrame
    df = pd.DataFrame(np.column_stack([xy, runs, dismissals]),
                      columns=['pitchX', 'pitchY', 'runs', 'dismissals'])
    df['x_binned'] = pd.cut(df.pitchX, x_edges)
    df['y_binned'] = pd.cut(df.pitchY, y_edges)

    # Group by bins and calculate total runs, dismissals, and count
    grouped = df.groupby(['x_binned', 'y_binned']).agg(
        total_runs=('runs', 'sum'),
        total_dismissals=('dismissals', 'sum'),
        count=('runs', 'count')
    ).reset_index()

    # Filter by minimum ball count
    grouped = grouped[grouped['count'] >= min_balls]

    # Calculate batting average (avoid division by zero)
    grouped['batting_average'] = grouped['total_runs'] / np.maximum(grouped['total_dismissals'], 1)

    # Normalize for color mapping
    mean_norm = (grouped['batting_average'] - grouped['batting_average'].min()) / \
                (grouped['batting_average'].max() - grouped['batting_average'].min())
    colours = plt.get_cmap(cmap)(mean_norm)
    grouped['colours'] = [tuple(x) for x in colours.tolist()]

    # Plot the heatmap
    fig = plt.figure()
    fig.set_size_inches(8, 5)
    fig.subplots_adjust(left=0,
                        right=1,
                        bottom=-0.2,
                        top=2) 
    ax = fig.add_subplot(111, projection='3d')

    # Add the wicket and pitch
    # plot_wicket_3d(ax, view='front')
    plot_wicket_3d(ax,
                   view='front',
                   pitch_colour=pitch_colour,
                   marking_colour=marking_colour,
                   outline_colour=outline_colour,
                   stump_colour=stump_colour,
                   wicket_colour=wicket_colour)

    # Add heatmap zones
    for row in grouped.itertuples():
        x = [row.x_binned.left, row.x_binned.right, row.x_binned.right, row.x_binned.left]
        y = [row.y_binned.left, row.y_binned.left, row.y_binned.right, row.y_binned.right]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(art3d.Poly3DCollection(verts,
                                                   edgecolors=outline_colour,
                                                   facecolors=row.colours,
                                                   alpha=0.9,
                                                   linewidths=0.5,
                                                   zorder=0))

    # Add legend
    legend_ypos = np.linspace(10, 0, 6)
    legend_labels = np.linspace(grouped['batting_average'].max(), grouped['batting_average'].min(), 6)
    legend_colours = plt.get_cmap(cmap)([y/10 for y in legend_ypos])
    for ypos, label, color in zip(legend_ypos, legend_labels, legend_colours):
        x = [-2.3, -2.6, -2.6, -2.3]
        y = [ypos, ypos, ypos+2, ypos+2]
        z = [0, 0, 0, 0]
        count = int(ypos/2)
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(art3d.Poly3DCollection(verts,
                                                   edgecolors=outline_colour,
                                                   facecolors=color,
                                                   alpha=0.9,
                                                   linewidths=0.5,
                                                   zorder=0))
        # ax.text(-2.1, ypos+1, 0, f'{label:.1f}', ha='left', va='center', fontsize=10)
        ax.text(x[0]+0.20, y[0]+1, z[0], f'{legend_labels[count]*1:.0f}', fontproperties=fp, size=12, c=outline_colour, ha='center', va='center')

    ax.text(-2.4,-1,0,legend_title, fontproperties=fp, size=17, c=outline_colour, ha='center', va='center')
    # Add titles and subtitles
    # add_title_axis(fig, title, subtitle_1, subtitle_2)
    ax.scatter(wickets[:, 0], wickets[:, 1], c='white', edgecolor='black', alpha=0.9, zorder=9)

    add_title_axis(fig,
                   title,
                   subtitle_1,
                   subtitle_2,
                   fp=fp,
                   title_colour=title_colour,
                   subtitle_colour=subtitle_colour)

    plt.show()

def bowling_strikerate_heatmap(xy,
                            total_balls,
                            dismissals,
                            wickets,
                            title='Bowling Strike Rate Heatmap',
                            subtitle_1='',
                            subtitle_2='',
                            legend_title='Strike Rate',
                            min_balls=10,
                            cmap='cool'):
    """
    Plots a heatmap showing batting average over a cricket pitch.

    ----------
    xy: A 2d array
        The x and y coordinates of the delivery pitching locations
    runs: A 1d array
        The number of runs scored for each ball
    dismissals: A 1d array
        Binary values indicating dismissals (1 for out, 0 for not out)
    title: A string
        The plot title
    subtitle_1: A string
        The plot's 1st subtitle
    subtitle_2: A string
        The plot's 2nd subtitle
    legend_title: A string
        The title of the value legend
    min_balls: An integer
        The minimum number of balls used for each heatmap zone
    cmap: Any valid matplotlib named colormap string
        The colour map used for the heatmap shading

    Returns
    -------
    matplotlib.axes.Axes
    """

    # Define some styling
    pitch_colour = 'white'
    wicket_colour = '#f5f6fa'
    marking_colour = '#595959'
    stump_colour = 'slategray'
    outline_colour = '#595959'
    title_colour = '#080a2e'
    subtitle_colour = '#9e9fa3'
    fname = './pitch_view/AlumniSans-SemiBold.ttf'
    fp = fm.FontProperties(fname=fname)

    # Define pitch bin edges
    XMIN, XMAX, XBIN = -1.83, 1.84, 0.2
    YMIN, YMAX, YBIN = 1, 15, 1
    x_edges = np.arange(XMIN, XMAX, XBIN)
    y_edges = np.arange(YMIN, YMAX, YBIN)

    # Combine data into a DataFrame
    df = pd.DataFrame(np.column_stack([xy, total_balls, dismissals]),
                      columns=['pitchX', 'pitchY', 'total_balls', 'dismissals'])
    df['x_binned'] = pd.cut(df.pitchX, x_edges)
    df['y_binned'] = pd.cut(df.pitchY, y_edges)

    # Group by bins and calculate total runs, dismissals, and count
    grouped = df.groupby(['x_binned', 'y_binned']).agg(
        total_balls=('total_balls', 'sum'),
        total_dismissals=('dismissals', 'sum'),
        count=('total_balls','count')
    ).reset_index()

    # Filter by minimum ball count
    grouped = grouped[grouped['count'] >= min_balls]

    # Calculate batting average (avoid division by zero)
    grouped['bowling_strikerate'] = grouped['total_balls'] / np.maximum(grouped['total_dismissals'], 1)

    # Normalize for color mapping
    mean_norm = (grouped['bowling_strikerate'] - grouped['bowling_strikerate'].min()) / \
                (grouped['bowling_strikerate'].max() - grouped['bowling_strikerate'].min())
    colours = plt.get_cmap(cmap)(mean_norm)
    grouped['colours'] = [tuple(x) for x in colours.tolist()]

    # Plot the heatmap
    fig = plt.figure()
    fig.set_size_inches(8, 5)
    fig.subplots_adjust(left=0,
                        right=1,
                        bottom=-0.2,
                        top=2) 
    ax = fig.add_subplot(111, projection='3d')

    # Add the wicket and pitch
    # plot_wicket_3d(ax, view='front')
    plot_wicket_3d(ax,
                   view='front',
                   pitch_colour=pitch_colour,
                   marking_colour=marking_colour,
                   outline_colour=outline_colour,
                   stump_colour=stump_colour,
                   wicket_colour=wicket_colour)

    # Add heatmap zones
    for row in grouped.itertuples():
        x = [row.x_binned.left, row.x_binned.right, row.x_binned.right, row.x_binned.left]
        y = [row.y_binned.left, row.y_binned.left, row.y_binned.right, row.y_binned.right]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(art3d.Poly3DCollection(verts,
                                                   edgecolors=outline_colour,
                                                   facecolors=row.colours,
                                                   alpha=0.9,
                                                   linewidths=0.5,
                                                   zorder=0))

    # Add legend
    legend_ypos = np.linspace(10, 0, 6)
    legend_labels = np.linspace(grouped['bowling_strikerate'].max(), grouped['bowling_strikerate'].min(), 6)
    legend_colours = plt.get_cmap(cmap)([y/10 for y in legend_ypos])
    for ypos, label, color in zip(legend_ypos, legend_labels, legend_colours):
        x = [-2.3, -2.6, -2.6, -2.3]
        y = [ypos, ypos, ypos+2, ypos+2]
        z = [0, 0, 0, 0]
        count = int(ypos/2)
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(art3d.Poly3DCollection(verts,
                                                   edgecolors=outline_colour,
                                                   facecolors=color,
                                                   alpha=0.9,
                                                   linewidths=0.5,
                                                   zorder=0))
        # ax.text(-2.1, ypos+1, 0, f'{label:.1f}', ha='left', va='center', fontsize=10)
        ax.text(x[0]+0.20, y[0]+1, z[0], f'{legend_labels[count]*1:.0f}', fontproperties=fp, size=12, c=outline_colour, ha='center', va='center')

    ax.text(-2.4,-1,0,legend_title, fontproperties=fp, size=17, c=outline_colour, ha='center', va='center')
    # Add titles and subtitles
    # add_title_axis(fig, title, subtitle_1, subtitle_2)
    ax.scatter(wickets[:, 0], wickets[:, 1], s=20, c='white', edgecolor='black', alpha=0.9, zorder=2)

    add_title_axis(fig,
                   title,
                   subtitle_1,
                   subtitle_2,
                   fp=fp,
                   title_colour=title_colour,
                   subtitle_colour=subtitle_colour)

    plt.show()

def pitch_map(dots_rh, runs_rh, boundaries_rh, wickets_rh, title='', subtitle_1='', subtitle_2=''):

    """ Plots a heatmap overlaid on wicket_3d front view, using a specified values array for square shading

    ----------
    xy: A 2d array
        The x and y coordinates of the delivery pitching locations
    title: A string
        The plot title
    subtitle_1: A string
        The plot's 1st subtitle
    subtitle_2: A string
        The plot's 2nd subtitle

    Returns
    -------
    matplotlib.axes.Axes"""

    # Define some styling
    # pitch_colour = 'white'
    # wicket_colour = '#f5f6fa'
    # marking_colour = 'cornflowerblue'
    # stump_colour = 'slategray'
    # outline_colour = 'lightsteelblue'
    # title_colour = '#080a2e'
    # subtitle_colour = '#9e9fa3'
    # fname = 'fonts/AlumniSans-SemiBold.ttf'
    # fp = fm.FontProperties(fname=fname)

    pitch_colour = 'white'
    wicket_colour = '#f5f6fa'
    marking_colour = '#595959'
    stump_colour = 'slategray'
    outline_colour = '#595959'
    title_colour = '#080a2e'
    subtitle_colour = '#9e9fa3'
    fname = './pitch_view/AlumniSans-SemiBold.ttf'
    fp = fm.FontProperties(fname=fname)

    X, Y, Z = get_density(dots_rh)

    # Plot a 2D KDE plot of the delivery pitch locations on a 3D pitch

    fig = plt.figure()
    fig.set_size_inches(8, 5)
    fig.subplots_adjust(left=0,
                        right=1,
                        bottom=-0.2,
                        top=2)  # Get rid of some excess whitespace - adjust to taste

    # ax = plt.gca(projection='3d')  # We'll plot on a 3D axis
    ax = fig.add_subplot(111, projection='3d')
    # We have data for a 3D surface plot, but we want to plot a 2D surface on the xy plane, so we'll set the Z axis
    # to zeros
    z_axis = np.zeros(X.shape)

    # We will manually set the colours of the surface based on the actual Z data using facecolors argument of
    # ax.plot_surface
    colours = plt.cm.PuRd(Z)

    # A trick to apply gradual alpha shading to the surface plot for cleaner looking visual

    for i in range(len(colours)):
        plane = colours[i]
        for j in range(len(plane)):
            row = plane[j]
            if (row[0:3].mean() >= 0.87) & (row[0:3].mean() < 0.92):
                row[3] = 0.25
            elif row[0:3].mean() >= 0.92:
                row[3] = 0

    # Plot the surfaces
    # ax.plot_surface(X,
    #                 Y,
    #                 z_axis,
    #                 cmap='Purples',
    #                 facecolors=colours,
    #                 linewidth=1,
    #                 antialiased=False)

    # Plot the pitch points as a scatter overlaid
    # ax.scatter(dots_rh[:, 0], dots_rh[:, 1], c='red', edgecolor='black', alpha=0.5, zorder=9)
    # ax.scatter(runs_rh[:, 0], runs_rh[:, 1], c='yellow', edgecolor='black', alpha=0.5, zorder=9)
    # ax.scatter(boundaries_rh[:, 0], boundaries_rh[:, 1], c='green', edgecolor='black', alpha=0.5, zorder=9)
    # ax.scatter(wickets_rh[:, 0], wickets_rh[:, 1], c='white', edgecolor='black', alpha=0.5, zorder=9)

    # Runs (Yellow)
    ax.scatter(runs_rh[:, 0], runs_rh[:, 1], 
            c='yellow', 
            edgecolor='#FF8C00',  # Dark orange hex code
            alpha=0.7, 
            zorder=9, 
            s=30, 
            linewidth=1.2, 
            marker='o',
            label="Runs")

    # Boundaries (Green)
    ax.scatter(boundaries_rh[:, 0], boundaries_rh[:, 1], 
            c='green', 
            edgecolor='#006400',  # Dark green hex code
            alpha=0.7, 
            zorder=10, 
            s=30, 
            linewidth=1.2, 
            marker='o',
            label="Boundaries")
    
    # Dots (Red)
    ax.scatter(dots_rh[:, 0], dots_rh[:, 1], 
            c='red', 
            edgecolor='#8B0000',  # Dark red hex code
            alpha=0.7, 
            zorder=11, 
            s=30, 
            linewidth=1.2, 
            marker='o',
            label="Dots")
    
    # Wickets (Blue)
    ax.scatter(wickets_rh[:, 0], wickets_rh[:, 1], 
           c='blue', 
           edgecolor='darkblue',  # Dark blue edge 
           alpha=0.7, 
           zorder=12, 
           s=30, 
           linewidth=1.2, 
           marker='o',
           label="Wickets") 
    
    # Add titles and subtitles
    add_title_axis(fig,
                   title,
                   subtitle_1,
                   subtitle_2,
                   fp=fp,
                   title_colour=title_colour,
                   subtitle_colour=subtitle_colour)

    # Generate a cricket pitch on the axis we created
    plot_wicket_3d(ax,
                   view='front',
                   pitch_colour=pitch_colour,
                   marking_colour=marking_colour,
                   outline_colour=outline_colour,
                   stump_colour=stump_colour,
                   wicket_colour=wicket_colour)
                   
    
    plt.show()

def average_turn_heatmap(xy,
                            turn,
                            wickets,
                            title='Average Turn Heatmap',
                            subtitle_1='',
                            subtitle_2='',
                            legend_title='Avg Turn',
                            min_balls=10,
                            cmap='Purples'):
    """
    Plots a heatmap showing batting average over a cricket pitch.

    ----------
    xy: A 2d array
        The x and y coordinates of the delivery pitching locations
    runs: A 1d array
        The number of runs scored for each ball
    dismissals: A 1d array
        Binary values indicating dismissals (1 for out, 0 for not out)
    title: A string
        The plot title
    subtitle_1: A string
        The plot's 1st subtitle
    subtitle_2: A string
        The plot's 2nd subtitle
    legend_title: A string
        The title of the value legend
    min_balls: An integer
        The minimum number of balls used for each heatmap zone
    cmap: Any valid matplotlib named colormap string
        The colour map used for the heatmap shading

    Returns
    -------
    matplotlib.axes.Axes
    """

    # Define some styling
    pitch_colour = 'white'
    wicket_colour = '#f5f6fa'
    marking_colour = '#595959'
    stump_colour = 'slategray'
    outline_colour = '#595959'
    title_colour = '#080a2e'
    subtitle_colour = '#9e9fa3'
    fname = './pitch_view/AlumniSans-SemiBold.ttf'
    fp = fm.FontProperties(fname=fname)

    # Define pitch bin edges
    XMIN, XMAX, XBIN = -1.2, 1.2, 0.2
    YMIN, YMAX, YBIN = 0, 15, 1
    x_edges = np.arange(XMIN, XMAX, XBIN)
    y_edges = np.arange(YMIN, YMAX, YBIN)

    # Combine data into a DataFrame
    df = pd.DataFrame(np.column_stack([xy, turn]),
                      columns=['pitchX', 'pitchY', 'turn'])
    df['x_binned'] = pd.cut(df.pitchX, x_edges)
    df['y_binned'] = pd.cut(df.pitchY, y_edges)

    # Group by bins and calculate total runs, dismissals, and count
    grouped = df.groupby(['x_binned', 'y_binned']).agg(
        total_turn=('turn', 'sum'),
        count=('turn', 'count')
    ).reset_index()

    # Filter by minimum ball count
    grouped = grouped[grouped['count'] >= min_balls]

    # Calculate batting average (avoid division by zero)
    grouped['average_turn'] = grouped['total_turn'] / np.maximum(grouped['count'], 1)

    cmap = plt.get_cmap('Purples').reversed()
    # Normalize for color mapping
    mean_norm = (grouped['average_turn'] - grouped['average_turn'].min()) / \
                (grouped['average_turn'].max() - grouped['average_turn'].min())
    colours = plt.get_cmap(cmap)(mean_norm)
    grouped['colours'] = [tuple(x) for x in colours.tolist()]

    # Plot the heatmap
    fig = plt.figure()
    fig.set_size_inches(8, 5)
    fig.subplots_adjust(left=0,
                        right=1,
                        bottom=-0.2,
                        top=2) 
    ax = fig.add_subplot(111, projection='3d')

    # Add the wicket and pitch
    # plot_wicket_3d(ax, view='front')
    plot_wicket_3d(ax,
                   view='front',
                   pitch_colour=pitch_colour,
                   marking_colour=marking_colour,
                   outline_colour=outline_colour,
                   stump_colour=stump_colour,
                   wicket_colour=wicket_colour)

    # Add heatmap zones
    for row in grouped.itertuples():
        x = [row.x_binned.left, row.x_binned.right, row.x_binned.right, row.x_binned.left]
        y = [row.y_binned.left, row.y_binned.left, row.y_binned.right, row.y_binned.right]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(art3d.Poly3DCollection(verts,
                                                   edgecolors=outline_colour,
                                                   facecolors=row.colours,
                                                   alpha=0.9,
                                                   linewidths=0.5,
                                                   zorder=0))

    # Add legend
    # max_turn = grouped['average_turn'].max()

    legend_ypos = np.linspace(10, 0, 6)
    legend_labels = np.linspace(grouped['average_turn'].max(), grouped['average_turn'].min(), 6)
    legend_colours = plt.get_cmap(cmap)([y/10 for y in legend_ypos])
    # legend_labels = np.linspace(1, 5, 5)
    # legend_colours = plt.get_cmap(cmap)(np.linspace(0, 1, len(legend_labels)))
    for ypos, label, color in zip(legend_ypos, legend_labels, legend_colours):
        x = [-2.3, -2.6, -2.6, -2.3]
        y = [ypos, ypos, ypos+2, ypos+2]
        z = [0, 0, 0, 0]
        count = int(ypos/2)
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(art3d.Poly3DCollection(verts,
                                                   edgecolors=outline_colour,
                                                   facecolors=color,
                                                   alpha=0.9,
                                                   linewidths=0.5,
                                                   zorder=0))
        # ax.text(-2.1, ypos+1, 0, f'{label:.1f}', ha='left', va='center', fontsize=10)
        ax.text(x[0]+0.20, y[0]+1, z[0], f'{legend_labels[count]*1:.1f}', fontproperties=fp, size=12, c=outline_colour, ha='center', va='center')
        # ax.text(x[0] + 0.20, y[0] + 1, z[0], f'{label:.0f}', fontproperties=fp, size=12, c=outline_colour, ha='center', va='center')

    ax.text(-2.4,-1,0,legend_title, fontproperties=fp, size=17, c=outline_colour, ha='center', va='center')
    # Add titles and subtitles
    # add_title_axis(fig, title, subtitle_1, subtitle_2)
    ax.scatter(wickets[:, 0], wickets[:, 1], c='white', edgecolor='black', alpha=0.9, zorder=9)

    add_title_axis(fig,
                   title,
                   subtitle_1,
                   subtitle_2,
                   fp=fp,
                   title_colour=title_colour,
                   subtitle_colour=subtitle_colour)

    plt.show()