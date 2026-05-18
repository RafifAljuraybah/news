# ── Keyword Density Over Time ──────────────────────────────────────────────
    st.divider()
    st.subheader("Policy vs Renewable Energy Keyword Density Over Time")
    st.caption(
        "3-month rolling average of adjusted keyword density (per 1,000 words), "
        "with linear trend lines showing overall direction of change."
    )

    if not non_outlier.empty:
        # Step 1 & 2: compute adjusted densities (already in filtered_articles as
        # adj_policy_density / adj_renewable_density — re-derive monthly averages)
        monthly_adj = (
            filtered_articles
            .groupby(pd.Grouper(key="published_date", freq="ME"))[
                ["adj_policy_density", "adj_renewable_density"]
            ]
            .mean()
        )

        # 3-month rolling smoothing
        monthly_adj["smooth_policy"] = (
            monthly_adj["adj_policy_density"].rolling(window=3, min_periods=1).mean()
        )
        monthly_adj["smooth_re"] = (
            monthly_adj["adj_renewable_density"].rolling(window=3, min_periods=1).mean()
        )

        # Linear trend lines via numpy polyfit
        x_numeric = np.arange(len(monthly_adj))  # integer indices for polyfit

        smooth_policy_filled = monthly_adj["smooth_policy"].fillna(0).values
        smooth_re_filled     = monthly_adj["smooth_re"].fillna(0).values

        z_policy = np.polyfit(x_numeric, smooth_policy_filled, 1)
        p_policy = np.poly1d(z_policy)
        trend_policy = p_policy(x_numeric)

        z_tech = np.polyfit(x_numeric, smooth_re_filled, 1)
        p_tech = np.poly1d(z_tech)
        trend_re = p_tech(x_numeric)

        # Percentage change along trend line (start → end)
        pol_pct  = ((trend_policy[-1] - trend_policy[0]) / trend_policy[0]  * 100) if trend_policy[0]  != 0 else 0
        tech_pct = ((trend_re[-1]     - trend_re[0])     / trend_re[0]      * 100) if trend_re[0]      != 0 else 0

        dates = monthly_adj.index

        fig_density_time = go.Figure()

        # Shaded fill between the two lines
        fig_density_time.add_trace(go.Scatter(
            x=list(dates) + list(dates[::-1]),
            y=list(monthly_adj["smooth_re"].fillna(0)) + list(monthly_adj["smooth_policy"].fillna(0)[::-1]),
            fill="toself",
            fillcolor="rgba(128,128,128,0.07)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name="Gap",
        ))

        # Policy smoothed line
        fig_density_time.add_trace(go.Scatter(
            x=dates, y=monthly_adj["smooth_policy"],
            mode="lines", name="Policy Density",
            line=dict(color=COLORS["Policy Focus"], width=2.5),
        ))

        # RE smoothed line
        fig_density_time.add_trace(go.Scatter(
            x=dates, y=monthly_adj["smooth_re"],
            mode="lines", name="RE Density",
            line=dict(color=COLORS["RE Focus"], width=2.5),
        ))

        # Policy trend line (dashed)
        fig_density_time.add_trace(go.Scatter(
            x=dates, y=trend_policy,
            mode="lines",
            name=f"Policy Trend ({pol_pct:+.1f}% change)",
            line=dict(color=COLORS["Policy Focus"], width=1.5, dash="dash"),
            opacity=0.8,
        ))

        # RE trend line (dashed)
        fig_density_time.add_trace(go.Scatter(
            x=dates, y=trend_re,
            mode="lines",
            name=f"RE Trend ({tech_pct:+.1f}% change)",
            line=dict(color=COLORS["RE Focus"], width=1.5, dash="dash"),
            opacity=0.8,
        ))

        fig_density_time.update_layout(
            title="Policy vs. Renewable Energy Used Terms Over Time",
            xaxis_title="Year",
            yaxis_title="Average Term Density (per 1,000 words)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="center", x=0.5,
            ),
        )

        st.plotly_chart(fig_density_time, use_container_width=True)

        # Summary callout
        pol_direction  = "increased" if pol_pct  > 0 else "decreased"
        tech_direction = "increased" if tech_pct > 0 else "decreased"
        st.info(
            f"Over the filtered period, **Policy** keyword density has "
            f"**{pol_direction} by {abs(pol_pct):.1f}%** and **Renewable Energy** "
            f"keyword density has **{tech_direction} by {abs(tech_pct):.1f}%** "
            f"(based on linear trend)."
        )
    else:
        st.info("No data available for the selected filters.")
