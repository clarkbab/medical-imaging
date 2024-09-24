from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest, norm, shapiro, yeojohnson
from statsmodels.graphics.gofplots import qqplot
from typing import List, Optional, Tuple

def plot_normality(
    x: List[float],
    figsize: Tuple[float, float] = (12, 4),
    pdf: Optional[PdfPages] = None,
    print_tests: bool = True,
    title: str = '') -> None:

    # Set figure title.
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title)

    # Plot sample and normal distribution.
    xs = np.linspace(np.min(x), np.max(x))
    mean = np.mean(x)
    std = np.std(x)
    ys = norm.pdf(xs, mean, std)
    axs[0][0].hist(x, density=True)
    axs[0][0].plot(xs, ys)

    # Create some normal data for a sanity check.
    normal_data = norm.rvs(loc=mean, scale=std, size=1000)

    # Plot Q-Q plot.
    qqplot(np.array(x), ax=axs[0][1], line='45', loc=mean, scale=std)

    # Transform data using Yeo-Johnson.
    xt, lamda = yeojohnson(x)
    xts = np.linspace(np.min(xt), np.max(xt))
    mean = np.mean(xt)
    std = np.std(xt)
    ys = norm.pdf(xts, mean, std)
    axs[1][0].hist(xt, density=True)
    axs[1][0].plot(xts, ys)

    # Plot transformed Q-Q plot.
    qqplot(np.array(xt), ax=axs[1][1], line='45', loc=mean, scale=std)

    # Perform hypothesis tests.
    shapiro_san = shapiro(normal_data)
    shapiro_res = shapiro(ys)
    ks_san = kstest(normal_data, norm.cdf, (mean, std))
    ks_res = kstest(x, norm.cdf)

    # Show and optionally save figure to PDF.
    plt.show()
    if pdf is not None:
        pdf.savefig(fig)

    if print_tests:
        print(f"""
        Shapiro-Wilkes:
            p-value: {shapiro_res.pvalue}

        Kolmogorov-Smirnov:
            p-value: {ks_res.pvalue}
        """)
