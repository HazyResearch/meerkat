import { rest } from 'msw';

const messages = {
    column_infos: [
        {
            name: "id",
            type: "number",
            cell_component: "Cell",
            cell_props: {},
            cell_data_prop: "data"
        },
        {
            name: "message",
            type: "string",
            cell_component: "Cell",
            cell_props: {},
            cell_data_prop: "data"
        },
        {
            name: "name",
            type: "string",
            cell_component: "Cell",
            cell_props: {},
            cell_data_prop: "data"
        },
        {
            name: "time",
            type: "string",
            cell_component: "Cell",
            cell_props: {},
            cell_data_prop: "data"
        },
        {
            name: "sender",
            type: "string",
            cell_component: "Cell",
            cell_props: {},
            cell_data_prop: "data"
        }
    ],
    posidxs: [0, 1, 2, 3],
    rows: [
        [0, "hello", "chatbot", "2021-01-01 00:00:00"],
        [1, "hi", "user", "2021-01-01 00:00:00"],
        [2, "how can i help", "chatbot", "2021-01-01 00:00:00"],
        [3, "im good thanks", "user", "2021-01-01 00:00:00"],
    ],
    full_length: 4,
    primary_key: "id"
}


// Define handlers that catch the corresponding requests and return the mock data.
export const handlers = [
    rest.post(`http://test.app/df/mock/rows`, (req, res, ctx) => {
        return res(ctx.status(200), ctx.json(messages))
    }),

    rest.get(`http://test.app/api/`, (req, res, ctx) => {
        return res(ctx.status(200), ctx.json({ "hello": "world" }))
    }),
];