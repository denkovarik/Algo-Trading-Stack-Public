import sys
import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os

def plot_interactive_candlestick(csv_filepath):
    # Read CSV, properly handling metadata rows
    data = pd.read_csv(csv_filepath, header=0, skiprows=[1, 2], parse_dates=[0], index_col=0)
    data.reset_index(inplace=True)
    data.rename(columns={data.columns[0]: 'Date'}, inplace=True)  # Explicitly name the date column

    fig = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])

    fig.update_layout(
        title=f"Interactive Candlestick Chart - {csv_filepath}",
        yaxis_title="Price",
        xaxis_title="Date",
        xaxis_rangeslider_visible=True,
        template="plotly_dark"
    )
    plot_dir = 'plots'
    filename = os.path.splitext(os.path.basename(csv_filepath))[0]
    html_file = f"{plot_dir}/{filename}_interactive_candlestick.html"
    fig.write_html(html_file)
    webbrowser.open('file://' + os.path.realpath(html_file))
    print(f"Interactive candlestick chart saved to {html_file}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)

    csv_filepath = sys.argv[1]
    plot_interactive_candlestick(csv_filepath)

