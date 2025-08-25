import pandas as pd

from typing import *

from mymi.typing import *
from mymi.utils import *

from .plotting import *

def plot_dataframe(
    data: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    ax: Optional[mpl.axes.Axes] = None,
    dpi: float = 300,
    exclude_x: Optional[Union[str, List[str]]] = None,
    figsize: Tuple[float, float] = (16, 6),
    filt: Optional[Dict[str, Any]] = {},
    fontsize: float = 6,
    fontsize_label: Optional[float] = None,
    fontsize_legend: Optional[float] = None,
    fontsize_stats: Optional[float] = None,
    fontsize_tick_label: Optional[float] = None,
    fontsize_title: Optional[float] = None,
    hue_connections_index: Optional[Union[str, List[str]]] = None,
    hue_hatch: Optional[Union[str, List[str]]] = None,
    hue_label: Optional[Union[str, List[str]]] = None,
    hue_order: Optional[List[str]] = None,
    hspace: Optional[float] = None,
    include_x: Optional[Union[str, List[str]]] = None,
    # legend_bbox: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = (1, 1),    # This is the point at which the legend anchor is anchored to the axes.
    # legend_borderaxespad: float = 1.0,
    legend_borderpad: float = 0.1,
    # legend_loc: str = 'upper left',     # This is actually the part of the legend that is anchored, not the legend's position within the axes.
    linecolour: str = 'black',
    line_width: float = 0.25,
    major_tick_freq: Optional[float] = None,
    marker_size: float = 4,
    minor_tick_freq: Optional[float] = None,
    n_cols: Optional[int] = None,
    n_rows: Optional[int] = None,
    outlier_legend_loc: str = 'upper left',
    palette: str = 'colorblind',
    savepath: Optional[str] = None,
    save_pad_inches: float = 0.03,
    share_y: bool = True,
    show_boxes: bool = True,
    show_hue_connections: bool = False,
    show_hue_connections_inliers: bool = False,
    show_legend: Union[bool, List[bool]] = True,
    show_points: bool = True,
    show_stats: bool = False,
    show_x_tick_labels: bool = True,
    show_x_tick_label_counts: bool = True,
    stats_alt: Literal['greater', 'less', 'two-sided'] = 'two-sided',
    stats_bar_alg_use_lowest_level: bool = True,
    stats_bar_alpha: float = 0.5,
    # Stats bars must sit at least this high above data points.
    stats_bar_grid_offset: float = 0.015,           # Proportion of window height.
    # This value is important! Without this number, the first grid line
    # will only have one bar, which means our bars will be stacked much higher. This allows
    # a little wiggle room.
    stats_bar_grid_offset_wiggle: float = 0.01,     # Proportion of window height.
    stats_bar_grid_spacing: float = 0.04,          # Proportion of window height.
    stats_bar_height: float = 0.007,                # Proportion of window height.
    stats_bar_show_direction: bool = False,
    stats_bar_text_offset: float = 0.008,            # Proportion of window height.
    stats_boot_df: Optional[pd.DataFrame] = None,
    stats_boot_df_cols: Optional[List[str]] = None,
    stats_exclude: List[str] = [],
    stats_exclude_left: List[str] = [],
    stats_exclude_right: List[str] = [],
    stats_paired: bool = True,
    stats_paired_by: Optional[Union[str, List[str]]] = None,
    stats_sig: List[float] = [0.05, 0.01, 0.001],
    style: Optional[Literal['box', 'violin']] = 'box',
    tick_length: float = 1.0,
    title: Optional[str] = None,
    title_x: Optional[float] = None,
    title_y: Optional[float] = None,
    x_label: Optional[str] = None,
    x_lim: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
    x_order: Optional[List[str]] = None,
    x_width: float = 0.8,
    x_tick_label: Optional[List[str]] = None,
    x_tick_label_rot: float = 0,
    y_label: Optional[str] = None,
    y_lim: Optional[Tuple[Optional[float], Optional[float]]] = (None, None)):
    if data is None:
        raise ValueError("'data' is None.")
    elif len(data) == 0:
        raise ValueError("DataFrame is empty.")
    hue_hatches = arg_to_list(hue_hatch, str)
    hue_labels = arg_to_list(hue_label, str)
    include_xs = arg_to_list(include_x, str)
    exclude_xs = arg_to_list(exclude_x, str)
    if show_hue_connections and hue_connections_index is None:
        raise ValueError(f"Please set 'hue_connections_index' to allow matching points between hues.")
    if show_stats and stats_paired_by is None:
        raise ValueError(f"Please set 'stats_paired_by' to determine sample pairing for Wilcoxon test.")
    x_tick_labels = arg_to_list(x_tick_label, str)

    # Set default fontsizes.
    if fontsize_label is None:
        fontsize_label = fontsize
    if fontsize_legend is None:
        fontsize_legend = fontsize
    if fontsize_stats is None:
        fontsize_stats = fontsize
    if fontsize_tick_label is None:
        fontsize_tick_label = fontsize
    if fontsize_title is None:
        fontsize_title = fontsize

    # Filter data.
    for k, v in filt.items():
        data = data[data[k] == v]
    if len(data) == 0:
        raise ValueError(f"DataFrame is empty after applying filters: {filt}.")
        
    # Include/exclude.
    if include_xs is not None:
        data = data[data[x].isin(include_xs)]
    if exclude_xs is not None:
        data = data[~data[x].isin(exclude_xs)]

    # Add outlier data.
    data = __add_outlier_info(data, x, y, hue)

    # Calculate global min/max values for when sharing y-limits across
    # rows (share_y=True).
    global_min_y = data[y].min()
    global_max_y = data[y].max()

    # Get x values.
    if x_order is None:
        x_order = list(sorted(data[x].unique()))

    # Determine x labels.
    groupby = x if hue is None else [x, hue]
    count_map = data.groupby(groupby)[y].count()
    if x_tick_labels is None:
        x_tick_labels = []
        for x_val in x_order:
            count = count_map.loc[x_val]
            if hue is not None:
                ns = count.values
                # Use a single number, e.g. (n=99) if all hues have the same number of points.
                if len(np.unique(ns)) == 1:
                    ns = ns[:1]
            else:
                ns = [count]
            label = f"{x_val}\n(n={','.join([str(n) for n in ns])})" if show_x_tick_label_counts else x_val
            x_tick_labels.append(label)

    # Create subplots if required.
    figsize = figsize_to_inches(figsize)
    if n_cols is None:
        n_cols = len(x_order)
    if n_rows is None:
        n_rows = int(np.ceil(len(x_order) / n_cols))
    if ax is not None:
        # Figure created externally - figsize handled.
        assert n_rows == 1
        axs = [ax]
        show_figure = False
    elif n_rows > 1:
        # Expand figsize height to match number of rows.
        figsize = (figsize[0], n_rows * figsize[1])

        # Use gridspec, this is required so we can create a final row that is smaller
        # than the other rows to preserve scale across rows. Sharing the x-axes doesn't work
        # here because the last one has smaller x-lim and will affect all rows - unless we use
        # inset axes? But this might be the reason our plotting was breaking.
        fig = plt.figure(dpi=dpi, figsize=figsize)
        hr = [1] * n_rows
        n_cols_final_row = len(x_order) % n_cols if len(x_order) % n_cols != 0 else n_cols
        width_ratios = [n_cols_final_row, n_cols - n_cols_final_row]
        gs = mpl.gridspec.GridSpec(n_rows, ncols=2, height_ratios=hr, hspace=hspace, width_ratios=width_ratios)

        # gridspec_kw = { 'hspace': hspace }
        # fig, axs = plt.subplots(n_rows, 1, constrained_layout=True, dpi=dpi, figsize=figsize, gridspec_kw=gridspec_kw, sharex=False, sharey=share_y)
        show_figure = True
    else:
        fig = plt.figure(dpi=dpi, figsize=figsize)
        axs = [plt.gca()]
        show_figure = True

    # Get hue order/colour/labels.
    if hue is not None:
        if hue_order is None:
            hue_order = list(sorted(data[hue].unique()))

        # Calculate x width for each hue.
        hue_width = x_width / len(hue_order) 

        # Create map from hue to colour.
        hue_palette = sns.color_palette(palette, n_colors=len(hue_order))
        hue_colours = dict((h, hue_palette[i]) for i, h in enumerate(hue_order))

        if hue_labels is not None:
            if len(hue_labels) != len(hue_order):
                raise ValueError(f"Length of 'hue_labels' ({hue_labels}) should match hues ({hue_order}).")
    
    # Expand args to match number of rows.
    if isinstance(show_legend, bool):
        show_legends = [show_legend] * n_rows
    else: 
        if len(show_legend) != n_rows:
            raise ValueError(f"Length of 'show_legend' ({len(show_legend)}) should match number of rows ({n_rows}).")
        else:
            show_legends = show_legend

    # Plot rows.
    for i, show_legend in zip(range(n_rows), show_legends):
        # Split data.
        row_x_order = x_order[i * n_cols:(i + 1) * n_cols]
        row_x_tick_labels = x_tick_labels[i * n_cols:(i + 1) * n_cols]

        # Get x colours.
        if hue is None:
            row_palette = sns.color_palette(palette, n_colors=len(row_x_order))
            x_colours = dict((x, row_palette[i]) for i, x in enumerate(row_x_order))

        # Get row data.
        row_df = data[data[x].isin(row_x_order)].copy()

        # Get x-axis limits.
        x_lim_row = list(x_lim)
        n_cols_row = len(row_x_order)
        if x_lim_row[0] is None:
            x_lim_row[0] = -0.5
        if x_lim_row[1] is None:
            x_lim_row[1] = n_cols_row - 0.5

        # Get y-axis limits.
        y_margin = 0.05
        row_min_y = row_df[y].min()
        row_max_y = row_df[y].max()
        min_y = global_min_y if share_y else row_min_y
        max_y = global_max_y if share_y else row_max_y
        y_lim_row = list(y_lim)
        if y_lim_row[0] is None:
            if y_lim_row[1] is None:
                width = max_y - min_y
                y_lim_row[0] = min_y - y_margin * width
                y_lim_row[1] = max_y + y_margin * width
            else:
                width = y_lim_row[1] - min_y
                y_lim_row[0] = min_y - y_margin * width
        else:
            if y_lim_row[1] is None:
                width = max_y - y_lim_row[0]
                y_lim_row[1] = max_y + y_margin * width

        # Get axis object.
        if n_rows > 1:
            # Create axis on gridspec.
            if i != n_rows - 1:
                ax = fig.add_subplot(gs[i, :])
            else:
                ax = fig.add_subplot(gs[i, 0])
        else:
            ax = axs[i]

        # Set axis limits.
        # This has to be done twice - once to set parent axes, and once to set child (inset) axes.
        # I can't remember why we made inset axes...?
        # Make this a function, as we need to call when stats bars exceed the y-axis limit.
        inset_ax = __set_axes_limits(ax, x_lim_row, y_lim_row)

        # Keep track of legend items.
        hue_artists = {}

        for j, x_val in enumerate(row_x_order):
            # Add x positions.
            if hue is not None:
                for k, hue_name in enumerate(hue_order):
                    x_pos = j - 0.5 * x_width + (k + 0.5) * hue_width
                    row_df.loc[(row_df[x] == x_val) & (row_df[hue] == hue_name), 'x_pos'] = x_pos
            else:
                x_pos = j
                row_df.loc[row_df[x] == x_val, 'x_pos'] = x_pos
                
            # Plot boxes.
            if show_boxes:
                if hue is not None:
                    for k, hue_name in enumerate(hue_order):
                        # Get hue data and pos.
                        hue_df = row_df[(row_df[x] == x_val) & (row_df[hue] == hue_name)]
                        if len(hue_df) == 0:
                            continue
                        hue_pos = hue_df.iloc[0]['x_pos']

                        # Get hue 'label' - allows us to use names more display-friendly than the data values.
                        hue_label = hue_name if hue_labels is None else hue_labels[k]

                        hatch = hue_hatches[k] if hue_hatches is not None else None
                        if style == 'box':
                            # Plot box.
                            res = inset_ax.boxplot(hue_df[y].dropna(), boxprops=dict(color=linecolour, facecolor=hue_colours[hue_name], linewidth=line_width), capprops=dict(color=linecolour, linewidth=line_width), flierprops=dict(color=linecolour, linewidth=line_width, marker='D', markeredgecolor=linecolour), medianprops=dict(color=linecolour, linewidth=line_width), patch_artist=True, positions=[hue_pos], showfliers=False, whiskerprops=dict(color=linecolour, linewidth=line_width), widths=hue_width)
                            if hatch is not None:
                                mpl.rcParams['hatch.linewidth'] = line_width
                                res['boxes'][0].set_hatch(hatch)
                                # res['boxes'][0].set(hatch=hatch)
                                # res['boxes'][0].set_edgecolor('white')
                                # res['boxes'][0].set(facecolor='white')

                            # Save reference to plot for legend.
                            if not hue_label in hue_artists:
                                hue_artists[hue_label] = res['boxes'][0]
                        elif style == 'violin':
                            # Plot violin.
                            res = inset_ax.violinplot(hue_df[y], positions=[hue_pos], widths=hue_width)

                            # Save reference to plot for legend.
                            if not hue_label in hue_artists:
                                hue_artists[hue_label] = res['boxes'][0]
                else:
                    # Plot box.
                    x_df = row_df[row_df[x] == x_val]
                    if len(x_df) == 0:
                        continue
                    x_pos = x_df.iloc[0]['x_pos']
                    if style == 'box':
                        inset_ax.boxplot(x_df[y], boxprops=dict(color=linecolour, facecolor=x_colours[x_val], linewidth=line_width), capprops=dict(color=linecolour, linewidth=line_width), flierprops=dict(color=linecolour, linewidth=line_width, marker='D', markeredgecolor=linecolour), medianprops=dict(color=linecolour, linewidth=line_width), patch_artist=True, positions=[x_pos], showfliers=False, whiskerprops=dict(color=linecolour, linewidth=line_width))
                    elif style == 'violin':
                        inset_ax.violinplot(x_df[y], positions=[x_pos])

            # Plot points.
            if show_points:
                if hue is not None:
                    for j, hue_name in enumerate(hue_order):
                        hue_df = row_df[(row_df[x] == x_val) & (row_df[hue] == hue_name)]
                        if len(hue_df) == 0:
                            continue
                        res = inset_ax.scatter(hue_df['x_pos'], hue_df[y], color=hue_colours[hue_name], edgecolors=linecolour, linewidth=line_width, s=marker_size, zorder=100)
                        if not hue_label in hue_artists:
                            hue_artists[hue_label] = res
                else:
                    x_df = row_df[row_df[x] == x_val]
                    inset_ax.scatter(x_df['x_pos'], x_df[y], color=x_colours[x_val], edgecolors=linecolour, linewidth=line_width, s=marker_size, zorder=100)

            # Identify connections between hues.
            if hue is not None and show_hue_connections:
                # Get column/value pairs to group across hue levels.
                # line_ids = row_df[(row_df[x] == x_val) & row_df['outlier']][outlier_cols]
                x_df = row_df[(row_df[x] == x_val)]
                if not show_hue_connections_inliers:
                    line_ids = x_df[x_df['outlier']][hue_connections_index]
                else:
                    line_ids = x_df[hue_connections_index]

                # Drop duplicates.
                line_ids = line_ids.drop_duplicates()

                # Get palette.
                line_palette = sns.color_palette('husl', n_colors=len(line_ids))

                # Plot lines.
                artists = []
                labels = []
                for j, (_, line_id) in enumerate(line_ids.iterrows()):
                    # Get line data.
                    x_df = row_df[(row_df[x] == x_val)]
                    for k, v in zip(line_ids.columns, line_id):
                        x_df = x_df[x_df[k] == v]
                    x_df = x_df.sort_values('x_pos')
                    x_pos = x_df['x_pos'].tolist()
                    y_data = x_df[y].tolist()

                    # Plot line.
                    lines = inset_ax.plot(x_pos, y_data, color=line_palette[j])

                    # Save line/label for legend.
                    artists.append(lines[0])
                    label = ':'.join(line_id.tolist())
                    labels.append(label)

                # Annotate outlier legend.
                if show_legend:
                    # Save main legend.
                    main_legend = inset_ax.get_legend()

                    # Show outlier legend.
                    inset_ax.legend(artists, labels, borderaxespad=legend_borderaxespad, borderpad=legend_borderpad, bbox_to_anchor=legend_bbox, fontsize=fontsize_legend, loc=outlier_legend_loc)

                    # Re-add main legend.
                    inset_ax.add_artist(main_legend)

        # Show legend.
        if hue is not None:
            if show_legend:
                # Filter 'hue_labels' based on hue 'artists'. Some hues may not be present in this row,
                # and 'hue_labels' is a global (across all rows) tracker.
                hue_labels = hue_order if hue_labels is None else hue_labels
                labels, artists = list(zip(*[(h, hue_artists[h]) for h in hue_labels if h in hue_artists]))

                # Show legend.
                # legend = inset_ax.legend(artists, labels, borderaxespad=legend_borderaxespad, borderpad=legend_borderpad, bbox_to_anchor=legend_bbox, fontsize=fontsize_legend, loc=legend_loc)
                # Calculate anchor point for 'upper left' of legend.
                legend_loc = 'upper left'
                bbox_to_anchor = (x_lim_row[1] + 0.05, y_lim_row[1])
                legend = inset_ax.legend(artists, labels, bbox_to_anchor=bbox_to_anchor, bbox_transform=inset_ax.transData, borderpad=legend_borderpad, fontsize=fontsize_legend, loc=legend_loc)
                frame = legend.get_frame()
                frame.set_boxstyle('square')
                frame.set_edgecolor('black')
                frame.set_linewidth(line_width)

        # Get pairs for stats tests.
        if show_stats:
            if hue is None:
                # Create pairs of 'x' values.
                if n_rows != 1:
                    raise ValueError(f"Can't show stats between 'x' values with multiple rows - not handled.")

                pairs = []
                max_skips = len(x_order) - 1
                for skip in range(1, max_skips + 1):
                    for j, x_val in enumerate(x_order):
                        other_x_idx = j + skip
                        if other_x_idx < len(x_order):
                            pair = (x_val, x_order[other_x_idx])
                            pairs.append(pair)
            else:
                # Create pairs of 'hue' values.
                pairs = []
                for x_val in row_x_order:
                    # Create pairs - start at lower numbers of skips as this will result in a 
                    # condensed plot.
                    hue_pairs = []
                    max_skips = len(hue_order) - 1
                    for skip in range(1, max_skips + 1):
                        for j, hue_val in enumerate(hue_order):
                            other_hue_index = j + skip
                            if other_hue_index < len(hue_order):
                                pair = (hue_val, hue_order[other_hue_index])
                                hue_pairs.append(pair)
                    pairs.append(hue_pairs)

        # Get p-values for each pair.
        if show_stats:
            nonsig_pairs = []
            nonsig_p_vals = []
            sig_pairs = []
            sig_p_vals = []

            if hue is None:
                for x_l, x_r in pairs:
                    row_pivot_df = row_df.pivot(index=stats_paired_by, columns=x, values=y).reset_index()
                    if x_l in row_pivot_df.columns and x_r in row_pivot_df.columns:
                        vals_l = row_pivot_df[x_l]
                        vals_r = row_pivot_df[x_r]

                        p_val = __calculate_p_val(vals_l, vals_r, stats_alt, stats_paired, stats_sig)

                        if p_val < stats_sig[0]:
                            sig_pairs.append((x_l, x_r))
                            sig_p_vals.append(p_val)
                        else:
                            nonsig_pairs.append((x_l, x_r))
                            nonsig_p_vals.append(p_val)
            else:
                for x_val, hue_pairs in zip(row_x_order, pairs):
                    x_df = row_df[row_df[x] == x_val]

                    hue_nonsig_pairs = []
                    hue_nonsig_p_vals = []
                    hue_sig_pairs = []
                    hue_sig_p_vals = []
                    for hue_l, hue_r in hue_pairs:
                        if stats_boot_df is not None:
                            # Load p-values from 'stats_boot_df'.
                            x_pivot_df = x_df.pivot(index=stats_paired_by, columns=[hue], values=[y]).reset_index()
                            if (y, hue_l) in x_pivot_df.columns and (y, hue_r) in x_pivot_df.columns:
                                vals_l = x_pivot_df[y][hue_l]
                                vals_r = x_pivot_df[y][hue_r]

                                # Don't add stats pair if main data is empty.
                                if len(vals_l) == 0 or len(vals_r) == 0:
                                    continue
                            else:
                                # Don't add stats pair if main data is empty.
                                continue
                            
                            # Get ('*', '<direction>') from dataframe. We have 'x_val' which is our region.
                            x_boot_df = stats_boot_df[stats_boot_df[x] == x_val]
                            boot_hue_l, boot_hue_r, boot_p_val = stats_boot_df_cols
                            x_pair_boot_df = x_boot_df[(x_boot_df[boot_hue_l] == hue_l) & (x_boot_df[boot_hue_r] == hue_r)]
                            if len(x_pair_boot_df) == 0:
                                raise ValueError(f"No matches found in 'stats_boot_df' for '{x}' ('{x_val}') '{boot_hue_l}' ('{hue_l}') and '{boot_hue_r}' ('{hue_r}').")
                            if len(x_pair_boot_df) > 1:
                                raise ValueError(f"Found multiple matches in 'stats_boot_df' for '{x}' ('{x_val}') '{boot_hue_l}' ('{hue_l}') and '{boot_hue_r}' ('{hue_r}').")
                            p_val = x_pair_boot_df.iloc[0][boot_p_val]

                            if p_val != '':
                                hue_sig_pairs.append((hue_l, hue_r))
                                hue_sig_p_vals.append(p_val)
                            else:
                                hue_nonsig_pairs.append((hue_l, hue_r))
                                hue_nonsig_p_vals.append(p_val)
                    else:
                        # Calculate p-values using stats tests.
                        for hue_l, hue_r in hue_pairs:
                            x_pivot_df = x_df.pivot(index=stats_paired_by, columns=[hue], values=[y]).reset_index()
                            if (y, hue_l) in x_pivot_df.columns and (y, hue_r) in x_pivot_df.columns:
                                vals_l = x_pivot_df[y][hue_l]
                                vals_r = x_pivot_df[y][hue_r]

                                p_val = __calculate_p_val(vals_l, vals_r, stats_alt, stats_paired, stats_sig)

                                if p_val < stats_sig[0]:
                                    hue_sig_pairs.append((hue_l, hue_r))
                                    hue_sig_p_vals.append(p_val)
                                else:
                                    hue_nonsig_pairs.append((hue_l, hue_r))
                                    hue_nonsig_p_vals.append(p_val)
                
                    nonsig_pairs.append(hue_nonsig_pairs)
                    nonsig_p_vals.append(hue_nonsig_p_vals)
                    sig_pairs.append(hue_sig_pairs)
                    sig_p_vals.append(hue_sig_p_vals)

        # Format p-values.
        if show_stats:
            if hue is None:
                sig_p_vals = __format_p_vals(sig_p_vals, stats_sig)
            else:
                sig_p_vals = [__format_p_vals(p, stats_sig) for p in sig_p_vals]

        # Remove 'excluded' pairs.
        if show_stats:
            filt_pairs = []
            filt_p_vals = []
            if hue is None:
                for (x_l, x_r), p_val in zip(sig_pairs, sig_p_vals):
                    if (x_l in stats_exclude or x_r in stats_exclude) or (x_l in stats_exclude_left) or (x_r in stats_exclude_right):
                        continue
                    filt_pairs.append((x_l, x_r))
                    filt_p_vals.append(p_val)
            else:
                for ps, p_vals in zip(sig_pairs, sig_p_vals):
                    hue_filt_pairs = []
                    hue_filt_p_vals = []
                    for (hue_l, hue_r), p_val in zip(ps, p_vals):
                        if (hue_l in stats_exclude or hue_r in stats_exclude) or (hue_l in stats_exclude_left) or (hue_r in stats_exclude_right):
                            continue
                        hue_filt_pairs.append((hue_l, hue_r))
                        hue_filt_p_vals.append(p_val)
                    filt_pairs.append(hue_filt_pairs)
                    filt_p_vals.append(hue_filt_p_vals)

        # Display stats bars.
        # To display stats bars, we fit a vertical grid over the plot and place stats bars
        # on the grid lines - so they look nice.
        if show_stats:
            # Calculate heights based on window height.
            y_height = max_y - min_y
            stats_bar_height = y_height * stats_bar_height
            stats_bar_grid_spacing = y_height * stats_bar_grid_spacing
            stats_bar_grid_offset = y_height * stats_bar_grid_offset
            stats_bar_grid_offset_wiggle = y_height * stats_bar_grid_offset_wiggle
            stats_bar_text_offset = y_height * stats_bar_text_offset
                
            if hue is None:
                # Calculate 'y_grid_offset' - the bottom of the grid.
                # For each pair, we calculate the max value of the data, as the bar should
                # lie above this. Then we find the smallest of these max values across all pairs.
                # 'stats_bar_grid_offset' is added to give spacing between the data and the bar.
                y_grid_offset = np.inf
                min_skip = None
                for x_l, x_r in filt_pairs:
                    if stats_bar_alg_use_lowest_level:
                        skip = x_order.index(x_r) - x_order.index(x_l) - 1
                        if min_skip is None:
                            min_skip = skip
                        elif skip > min_skip:
                            continue

                    x_l_df = row_df[row_df[x] == x_l]
                    x_r_df = row_df[row_df[x] == x_r]
                    y_max = max(x_l_df[y].max(), x_r_df[y].max())
                    y_max = y_max + stats_bar_grid_offset
                    if y_max < y_grid_offset:
                        y_grid_offset = y_max

                # Add data offset.
                y_grid_offset = y_grid_offset + stats_bar_grid_offset_wiggle

                # Annotate figure.
                # We keep track of bars we've plotted using 'y_idxs'.
                # This is a mapping from the hue to the grid positions that have already
                # been used for either a left or right hand side of a bar.
                y_idxs: Dict[str, List[int]] = {}
                for (x_l, x_r), p_val in zip(filt_pairs, filt_p_vals):
                    # Get plot 'x_pos' for each x value.
                    x_l_df = row_df[row_df[x] == x_l]
                    x_r_df = row_df[row_df[x] == x_r]
                    x_left = x_l_df['x_pos'].iloc[0]
                    x_right = x_r_df['x_pos'].iloc[0]

                    # Get 'y_idx_min' (e.g. 0, 1, 2,...) which tells us the lowest grid line
                    # we can use based on our data points.
                    # We calculate this by finding the max data value for the pair, and also
                    # the max values for any hues between the pair values - as our bar should
                    # not collide with these 'middle' hues. 
                    y_data_maxes = [x_end_df[y].max() for x_end_df in [x_l_df, x_r_df]]
                    n_mid_xs = x_order.index(x_r) - x_order.index(x_l) - 1
                    for j in range(n_mid_xs):
                        x_mid = x_order[x_order.index(x_l) + j + 1]
                        x_mid_df = row_df[row_df[x] == x_mid]
                        y_data_max = x_mid_df[y].max()
                        y_data_maxes.append(y_data_max)
                    y_data_max = max(y_data_maxes) + stats_bar_grid_offset
                    y_idx_min = int(np.ceil((y_data_max - y_grid_offset) / stats_bar_grid_spacing))

                    # We don't want our new stats bar to collide with any existing bars.
                    # Get the y positions for all stats bar that have already been plotted
                    # and that have their left or right end at one of the 'middle' hues for
                    # our current pair.
                    n_mid_xs = x_order.index(x_r) - x_order.index(x_l) - 1
                    y_idxs_mid_xs = []
                    for j in range(n_mid_xs):
                        x_mid = x_order[x_order.index(x_l) + j + 1]
                        if x_mid in y_idxs:
                            y_idxs_mid_xs += y_idxs[x_mid]

                    # Get the next free position that doesn't collide with any existing bars.
                    y_idx_max = 100
                    for y_idx in range(y_idx_min, y_idx_max):
                        if y_idx not in y_idxs_mid_xs:
                            break

                    # Plot bar.
                    y_min = y_grid_offset + y_idx * stats_bar_grid_spacing
                    y_max = y_min + stats_bar_height
                    inset_ax.plot([x_left, x_left, x_right, x_right], [y_min, y_max, y_max, y_min], alpha=stats_bar_alpha, color=linecolour, linewidth=line_width)    

                    # Adjust y-axis limits if bar would be plotted outside of window.
                    # Unless y_lim is set manually.
                    y_lim_top = y_max + 1.5 * stats_bar_grid_spacing
                    if y_lim[1] is None and y_lim_top > y_lim_row[1]:
                        y_lim_row[1] = y_lim_top
                        inset_ax = __set_axes_limits(axs[i], x_lim_row, y_lim_row, inset_ax=inset_ax)

                    # Plot p-value.
                    x_text = (x_left + x_right) / 2
                    y_text = y_max + stats_bar_text_offset
                    inset_ax.text(x_text, y_text, p_val, alpha=stats_bar_alpha, fontsize=fontsize_stats, horizontalalignment='center', verticalalignment='center')

                    # Save position of plotted stats bar.
                    if not x_l in y_idxs:
                        y_idxs[x_l] = [y_idx]
                    elif y_idx not in y_idxs[x_l]:
                        y_idxs[x_l] = list(sorted(y_idxs[x_l] + [y_idx]))
                    if not x_r in y_idxs:
                        y_idxs[x_r] = [y_idx]
                    elif y_idx not in y_idxs[x_r]:
                        y_idxs[x_r] = list(sorted(y_idxs[x_r] + [y_idx]))
            else:
                for hue_pairs, hue_p_vals in zip(filt_pairs, filt_p_vals):
                    # Calculate 'y_grid_offset' - the bottom of the grid.
                    # For each pair, we calculate the max value of the data, as the bar should
                    # lie above this. Then we find the smallest of these max values across all pairs.
                    # 'stats_bar_grid_offset' is added to give spacing between the data and the bar.
                    y_grid_offset = np.inf
                    min_skip = None
                    for hue_l, hue_r in hue_pairs:
                        if stats_bar_alg_use_lowest_level:
                            skip = hue_order.index(hue_r) - hue_order.index(hue_l) - 1
                            if min_skip is None:
                                min_skip = skip
                            elif skip > min_skip:
                                continue

                        hue_l_df = x_df[x_df[hue] == hue_l]
                        hue_r_df = x_df[x_df[hue] == hue_r]
                        y_max = max(hue_l_df[y].max(), hue_r_df[y].max())
                        y_max = y_max + stats_bar_grid_offset
                        if y_max < y_grid_offset:
                            y_grid_offset = y_max

                    # Add data offset.
                    y_grid_offset = y_grid_offset + stats_bar_grid_offset_wiggle

                    # Annotate figure.
                    # We keep track of bars we've plotted using 'y_idxs'.
                    # This is a mapping from the hue to the grid positions that have already
                    # been used for either a left or right hand side of a bar.
                    y_idxs: Dict[str, List[int]] = {}
                    for (hue_l, hue_r), p_val in zip(hue_pairs, hue_p_vals):
                        # Get plot 'x_pos' for each hue.
                        hue_l_df = x_df[x_df[hue] == hue_l]
                        hue_r_df = x_df[x_df[hue] == hue_r]
                        if len(hue_l_df) == 0 or len(hue_r_df) == 0:
                            continue
                        x_left = hue_l_df['x_pos'].iloc[0]
                        x_right = hue_r_df['x_pos'].iloc[0]

                        # Get 'y_idx_min' (e.g. 0, 1, 2,...) which tells us the lowest grid line
                        # we can use based on our data points.
                        # We calculate this by finding the max data value for the pair, and also
                        # the max values for any hues between the pair values - as our bar should
                        # not collide with these 'middle' hues. 
                        y_data_maxes = [hue_df[y].max() for hue_df in [hue_l_df, hue_r_df]]
                        n_mid_hues = hue_order.index(hue_r) - hue_order.index(hue_l) - 1
                        for j in range(n_mid_hues):
                            hue_mid = hue_order[hue_order.index(hue_l) + j + 1]
                            hue_mid_df = x_df[x_df[hue] == hue_mid]
                            y_data_max = hue_mid_df[y].max()
                            y_data_maxes.append(y_data_max)
                        y_data_max = max(y_data_maxes) + stats_bar_grid_offset
                        y_idx_min = int(np.ceil((y_data_max - y_grid_offset) / stats_bar_grid_spacing))

                        # We don't want our new stats bar to collide with any existing bars.
                        # Get the y positions for all stats bar that have already been plotted
                        # and that have their left or right end at one of the 'middle' hues for
                        # our current pair.
                        n_mid_hues = hue_order.index(hue_r) - hue_order.index(hue_l) - 1
                        y_idxs_mid_hues = []
                        for j in range(n_mid_hues):
                            hue_mid = hue_order[hue_order.index(hue_l) + j + 1]
                            if hue_mid in y_idxs:
                                y_idxs_mid_hues += y_idxs[hue_mid]

                        # Get the next free position that doesn't collide with any existing bars.
                        y_idx_max = 100
                        for y_idx in range(y_idx_min, y_idx_max):
                            if y_idx not in y_idxs_mid_hues:
                                break

                        # Plot bar.
                        y_min = y_grid_offset + y_idx * stats_bar_grid_spacing
                        y_max = y_min + stats_bar_height
                        inset_ax.plot([x_left, x_left, x_right, x_right], [y_min, y_max, y_max, y_min], alpha=stats_bar_alpha, color=linecolour, linewidth=line_width)    

                        # Adjust y-axis limits if bar would be plotted outside of window.
                        # Unless y_lim is set manually.
                        y_lim_top = y_max + 1.5 * stats_bar_grid_spacing
                        if y_lim[1] is None and y_lim_top > y_lim_row[1]:
                            y_lim_row[1] = y_lim_top
                            inset_ax = __set_axes_limits(axs[i], x_lim_row, y_lim_row, inset_ax=inset_ax)

                        # Plot p-value.
                        x_text = (x_left + x_right) / 2
                        y_text = y_max + stats_bar_text_offset
                        inset_ax.text(x_text, y_text, p_val, alpha=stats_bar_alpha, fontsize=fontsize_stats, horizontalalignment='center', verticalalignment='center')

                        # Save position of plotted stats bar.
                        if not hue_l in y_idxs:
                            y_idxs[hue_l] = [y_idx]
                        elif y_idx not in y_idxs[hue_l]:
                            y_idxs[hue_l] = list(sorted(y_idxs[hue_l] + [y_idx]))
                        if not hue_r in y_idxs:
                            y_idxs[hue_r] = [y_idx]
                        elif y_idx not in y_idxs[hue_r]:
                            y_idxs[hue_r] = list(sorted(y_idxs[hue_r] + [y_idx]))
          
        # Set axis labels.
        x_label = x_label if x_label is not None else ''
        y_label = y_label if y_label is not None else ''
        inset_ax.set_xlabel(x_label, fontsize=fontsize_label)
        inset_ax.set_ylabel(y_label, fontsize=fontsize_label)
                
        # Set axis tick labels.
        inset_ax.set_xticks(list(range(len(row_x_tick_labels))))
        if show_x_tick_labels:
            inset_ax.set_xticklabels(row_x_tick_labels, fontsize=fontsize_tick_label, rotation=x_tick_label_rot)
        else:
            inset_ax.set_xticklabels([])

        inset_ax.tick_params(axis='y', which='major', labelsize=fontsize_tick_label)

        # Set y axis major ticks.
        if major_tick_freq is not None:
            major_tick_min = y_lim[0]
            if major_tick_min is None:
                major_tick_min = inset_ax.get_ylim()[0]
            major_tick_max = y_lim[1]
            if major_tick_max is None:
                major_tick_max = inset_ax.get_ylim()[1]
            
            # Round range to nearest multiple of 'major_tick_freq'.
            major_tick_min = np.ceil(major_tick_min / major_tick_freq) * major_tick_freq
            major_tick_max = np.floor(major_tick_max / major_tick_freq) * major_tick_freq
            n_major_ticks = int((major_tick_max - major_tick_min) / major_tick_freq) + 1
            major_ticks = np.linspace(major_tick_min, major_tick_max, n_major_ticks)
            major_tick_labels = [str(round(t, 3)) for t in major_ticks]     # Some weird str() conversion without rounding.
            inset_ax.set_yticks(major_ticks)
            inset_ax.set_yticklabels(major_tick_labels)

        # Set y axis minor ticks.
        if minor_tick_freq is not None:
            minor_tick_min = y_lim[0]
            if minor_tick_min is None:
                minor_tick_min = inset_ax.get_ylim()[0]
            minor_tick_max = y_lim[1]
            if minor_tick_max is None:
                minor_tick_max = inset_ax.get_ylim()[1]
            
            # Round range to nearest multiple of 'minor_tick_freq'.
            minor_tick_min = np.ceil(minor_tick_min / minor_tick_freq) * minor_tick_freq
            minor_tick_max = np.floor(minor_tick_max / minor_tick_freq) * minor_tick_freq
            n_minor_ticks = int((minor_tick_max - minor_tick_min) / minor_tick_freq) + 1
            minor_ticks = np.linspace(minor_tick_min, minor_tick_max, n_minor_ticks)
            inset_ax.set_yticks(minor_ticks, minor=True)

        # Set y grid lines.
        inset_ax.grid(axis='y', alpha=0.1, color='grey', linewidth=line_width)
        inset_ax.set_axisbelow(True)

        # Set axis spine/tick linewidths and tick lengths.
        spines = ['top', 'bottom','left','right']
        for spine in spines:
            inset_ax.spines[spine].set_linewidth(line_width)
        inset_ax.tick_params(which='both', length=tick_length, width=line_width)

    # Set title.
    okwargs = dict(
        fontsize=fontsize_title,
        style='italic',
    )
    fig.suptitle(title, **okwargs)

    # Save plot to disk.
    if savepath is not None:
        savepath = escape_filepath(savepath)
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        fig.savefig(savepath, bbox_inches='tight', pad_inches=0.03)
        logging.info(f"Saved plot to '{savepath}'.")

    # Show plot.
    if show_figure:
        fig.show()

def plot_histogram(
    data: Union[ImageArray3D, str],
    ax: Optional[mpl.axes.Axes] = None,
    diff: Optional[ImageArray3D] = None,
    fontsize: float = 12,
    n_bins: int = 100,
    title: Optional[str] = None,
    x_lim: Optional[Tuple[Optional[float], Optional[float]]] = None) -> None:
    # Handle arguments.
    if isinstance(data, str):
        if data.endswith('.nii.gz'):
            data, _, _ = load_nifti(data)
    if ax is None:
        ax = plt.gca()
        show = True
    else:
        show = False
    
    # Plot histogram.
    if diff is not None:
        min_val = np.min([data.min(), diff.min()])
        max_val = np.max([data.max(), diff.max()])
    else:
        min_val, max_val = data.min(), data.max()
    bins = np.linspace(min_val, max_val, n_bins)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(data, bins=bins)
    if diff is not None:
        diff_hist, _ = np.histogram(diff, bins=bins)
        hist = hist - diff_hist
    colours = ['blue' if h >= 0 else 'red' for h in hist]
    ax.bar(bin_centres, hist, width=np.diff(bins), color=colours, edgecolor='black')
    ax.set_xlabel('Value', fontsize=fontsize)
    ax.set_ylabel('Count', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    # Add stats box.
    # Calculate stats on binned values to allow for diff calc.
    if diff is None:
        mean, std, median = data.mean(), data.std(), np.median(data)
    else:
        mean = np.sum(np.abs(hist) * bin_centres) / np.abs(hist).sum()
        std = np.sqrt(np.sum(np.abs(hist) * (bin_centres - mean) ** 2) / np.sum(np.abs(hist)))
        # Interpolate median value.
        com = np.sum(np.abs(hist) * np.arange(len(hist))) / np.sum(np.abs(hist))
        com_floor, com_ceil = int(np.floor(com)), int(np.ceil(com))
        median = bin_centres[com_floor] + (com_ceil - com) * (bin_centres[com_ceil] - bin_centres[com_floor])
    text = f"""\
mean: {mean:.2f}\n\
std: {std:.2f}\n\
median: {median:.2f}\n\
min/max: {min_val:.2f}/{max_val:.2f}\
"""
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, text, bbox=props, fontsize=fontsize, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')

    if show:
        plt.show()

def plot_histograms(
    datas: Union[ImageArray3D, List[ImageArray3D]],
    axs: Optional[Union[mpl.axes.Axes, List[mpl.axes.Axes]]] = None,
    diffs: Optional[Union[ImageArray3D, List[ImageArray3D]]] = None,
    figsize: Tuple[float, float] = (6, 4),
    **kwargs) -> None:
    datas = arg_to_list(datas, (ImageArray3D, str))
    diffs = arg_to_list(diffs, (ImageArray3D, None), broadcast=len(datas))
    assert len(diffs) == len(datas)
    figsize = (len(datas) * figsize[0], figsize[1])
    if axs is None:
        _, axs = plt.subplots(1, len(datas), figsize=figsize, squeeze=False)
        axs = axs[0]    # Only one row.
        show = True
    else:
        axs = arg_to_list(axs, mpl.axes.Axes)
        assert len(axs) == len(datas), "Number of axes must match number of data arrays."
        show = False
    for a, d, diff in zip(axs, datas, diffs):
        plot_histogram(d, ax=a, diff=diff, **kwargs)

    if show:
        plt.show()

def __add_outlier_info(df, x, y, hue):
    if hue is not None:
        groupby = [hue, x]
    else:
        groupby = x
    q1_map = df.groupby(groupby)[y].quantile(.25)
    q3_map = df.groupby(groupby)[y].quantile(.75)
    def q_func_build(qmap):
        def q_func(row):
            if isinstance(groupby, list):
                key = tuple(row[groupby])
            else:
                key = row[groupby]
            return qmap[key]
        return q_func
    q1 = df.apply(q_func_build(q1_map), axis=1)
    df = df.assign(q1=q1)
    df = df.assign(q3=df.apply(q_func_build(q3_map), axis=1))
    df = df.assign(iqr=df.q3 - df.q1)
    df = df.assign(outlier_lim_low=df.q1 - 1.5 * df.iqr)
    df = df.assign(outlier_lim_high=df.q3 + 1.5 * df.iqr)
    df = df.assign(outlier=(df[y] < df.outlier_lim_low) | (df[y] > df.outlier_lim_high))
    return df

def __calculate_p_val(
    a: List[float],
    b: List[float],
    stats_alt: Literal['greater', 'less', 'two-sided'],
    stats_paired: bool,
    stats_sig: List[float]) -> float:

    if stats_paired:
        if np.any(a.isna()) or np.any(b.isna()):
            raise ValueError(f"Unpaired data... add more info.")

        # Can't calculate paired test without differences.
        if np.all(b - a == 0):
            raise ValueError(f"Paired data is identical ... add more info.")
    else:
        # Remove 'nan' values.
        a = a[~a.isna()]
        b = b[~b.isna()]

    # Check for presence of data.
    if len(a) == 0 or len(b) == 0:
        raise ValueError(f"Empty data... add more info.")

    # Calculate p-value.
    if stats_paired:
        _, p_val = scipy.stats.wilcoxon(a, b, alternative=stats_alt)
    else:
        _, p_val = scipy.stats.mannwhitneyu(a, b, alternative=stats_alt)

    return p_val

def __format_p_vals(
    p_vals: List[float],
    stats_sig: List[float]) -> List[str]:
    p_vals_f = []
    for p in p_vals:
        p_f = ''
        for s in stats_sig:
            if p < s:
                p_f += '*'
            else:
                break
        p_vals_f.append(p_f)
    return p_vals_f

def __set_axes_limits(
    parent_ax: mpl.axes.Axes,
    x_lim: Tuple[float, float],
    y_lim: Tuple[float, float],
    inset_ax: Optional[mpl.axes.Axes] = None) -> mpl.axes.Axes:
    # Set parent axes limits.
    parent_ax.set_xlim(*x_lim)
    parent_ax.set_ylim(*y_lim)
    # parent_ax.axis('off')

    # Create inset if not passed.
    if inset_ax is None:
        # Don't create inset for now.
        # inset_ax = parent_ax.inset_axes([x_lim[0], y_lim[0], x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]], transform=parent_ax.transData)
        inset_ax = parent_ax

    # Set inset axis limits.
    inset_ax.set_xlim(*x_lim)
    inset_ax.set_ylim(*y_lim)

    return inset_ax
