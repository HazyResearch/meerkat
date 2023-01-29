// src/mocks/handlers.ts
import { DataFrameChunk } from '$lib/api/dataframe';
import { API_URL } from '$lib/constants';
import { rest } from 'msw'
import { get } from 'svelte/store';

// Mock Data


// /**
//  * A chunk of a DataFrame holds the data for a subset of the rows and columns in a full
//  * DataFrame that lives on the backend. 
//  */
// export class DataFrameChunk {

//     column_infos: Array<ColumnInfo>
//     columns: Array<string>
//     posidxs: Array<number>
//     keyidxs: Array<string>
//     rows: Array<Array<any>>
//     full_length: number
//     primary_key: string

//     constructor(
//         column_infos: Array<ColumnInfo>,
//         posidxs: Array<number>,
//         rows: Array<Array<any>>,
//         full_length: number,
//         primary_key: string
//     ) {
//         this.column_infos = column_infos;
//         this.columns = this.column_infos.map((col: any) => col.name);
//         this.posidxs = posidxs;
//         this.rows = rows;
//         this.full_length = full_length;
//         this.primary_key = primary_key

//         let primary_key_index = this.columns.findIndex((c) => c === primary_key);
//         this.keyidxs = this.rows.map((row) => row[primary_key_index])
//     }

//     get_cell(row: number, column: string) {
//         let column_idx = this.columns.indexOf(column);
//         let column_info = this.column_infos[column_idx];
//         return {
//             data: this.rows[row][this.columns.indexOf(column)],
//             cell_component: column_info.cell_component,
//             cell_props: column_info.cell_props,
//             cell_data_prop: column_info.cell_data_prop,
//             column: column
//         }
//     }
//     get_column(column: string) {
//         let column_idx = this.columns.indexOf(column);
//         let column_info = this.column_infos[column_idx];
//         return {
//             data: this.rows.map((row) => row[column_idx]),
//             cell_component: column_info.cell_component,
//             cell_props: column_info.cell_props,
//             cell_data_prop: column_info.cell_data_prop,
//             column: column
//         }
//     }

//     length() {
//         return this.rows.length;
//     }

// }

// Messages as a DataFrameChunk
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