import { post } from '$lib/utils/requests';

export interface ColumnInfo {
    name: string;
    type: string;
    cell_component: string;
    cell_props: any
}

export interface DataPanelSchema {
    columns: Array<ColumnInfo>;
    id: string;
}
export interface DataPanelRows {
    column_infos: Array<ColumnInfo>
    indices: Array<number>
    rows: Array<Array<any>>
    full_length: number
}

export async function get_schema(
    api_url: string, datapanel_id: string, columns: Array<string> | null = null
): Promise<DataPanelSchema> {
    return await post(`${api_url}/dp/${datapanel_id}/schema`, { columns: columns });
}

export async function get_rows(api_url: string, datapanel_id: string, start: number, end: number): Promise<DataPanelRows> {
    return await post(`${api_url}/dp/${datapanel_id}/rows`, { start: start, end: end });
}


export async function create_column(
    api_url: string, datapanel_id: string, text: string
): Promise<string> {
    return await post(`${api_url}/dp/${datapanel_id}/create_column`, { text: text });
}