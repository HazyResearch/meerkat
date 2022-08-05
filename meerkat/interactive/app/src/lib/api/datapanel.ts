import { post } from '$lib/utils/requests';

export interface ColumnInfo {
    name: string;
    type: string;
    cell_component: string;
    cell_props: any
}


export interface DataPanelRows {
    column_infos: Array<ColumnInfo>
    indices: Array<number>
    rows: Array<Array<any>>
    full_length: number
}


export async function get_rows(api_url: string, dp_id: string, start: number, end: number): Promise<any> {
    console.log("hello in datapanel.ts");
    console.log(`${api_url}/dp/${dp_id}/rows`);
    let data_promise = await post(`${api_url}/dp/${dp_id}/rows`, { start: start, end: end });
    return data_promise;
}
