import streamlit as st
import pandas as pd


def main():
    df = pd.read_parquet("market_data/NKE/options_20241128.parquet")
    st.dataframe(df)  # Interactive table
    st.line_chart(df[["ask_price", "bid_price"]])


if __name__ == "__main__":
    main()
