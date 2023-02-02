import { render } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import Vega from './Vega.svelte';

describe('Vega', () => {
    it('should render vega', async () => {
        const { container } = render(Vega, {
            props: {
                data: {
                    table: [
                        { category: "A", amount: 28 },
                        { category: "B", amount: 55 },
                        { category: "CC", amount: 43 },
                        { category: "D", amount: 91 },
                        { category: "E", amount: 81 },
                        { category: "F", amount: 53 },
                        { category: "G", amount: 19 },
                        { category: "H", amount: 87 },
                    ]
                },
                spec: {
                    $schema: "https://vega.github.io/schema/vega/v5.json",
                    width: 400,
                    height: 200,
                    padding: { left: 5, right: 5, top: 5, bottom: 5 },
                    data: [
                        {
                            name: "table",
                        },
                    ],
                    signals: [
                        {
                            name: "tooltip",
                            value: {},
                            on: [
                                { events: "rect:mouseover", update: "datum" },
                                { events: "rect:mouseout", update: "{}" },
                            ],
                        },
                    ],
                    scales: [
                        {
                            name: "xscale",
                            type: "band",
                            domain: { data: "table", field: "category" },
                            range: "width",
                        },
                        {
                            name: "yscale",
                            domain: { data: "table", field: "amount" },
                            nice: true,
                            range: "height",
                        },
                    ],
                    axes: [
                        { orient: "bottom", scale: "xscale" },
                        { orient: "left", scale: "yscale" },
                    ],
                    marks: [
                        {
                            type: "rect",
                            from: { data: "table" },
                            encode: {
                                enter: {
                                    x: { scale: "xscale", field: "category", offset: 1 },
                                    width: { scale: "xscale", band: 1, offset: -1 },
                                    y: { scale: "yscale", field: "amount" },
                                    y2: { scale: "yscale", value: 0 },
                                },
                                update: {
                                    fill: { value: "steelblue" },
                                },
                                hover: {
                                    fill: { value: "red" },
                                },
                            },
                        },
                        {
                            type: "text",
                            encode: {
                                enter: {
                                    align: { value: "center" },
                                    baseline: { value: "bottom" },
                                    fill: { value: "#333" },
                                },
                                update: {
                                    x: { scale: "xscale", signal: "tooltip.category", band: 0.5 },
                                    y: { scale: "yscale", signal: "tooltip.amount", offset: -2 },
                                    text: { signal: "tooltip.amount" },
                                    fillOpacity: [
                                        { test: "datum === tooltip", value: 0 },
                                        { value: 1 },
                                    ],
                                },
                            },
                        },
                    ],
                }
            },
        });

        expect(container).toMatchSnapshot();
    });
});