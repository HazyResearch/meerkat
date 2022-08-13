import { get, post } from '$lib/utils/requests';
import type { DataPanelRows } from '$lib/api/datapanel';

export type SliceKey = string | number 

export interface SliceByInfo {
    id: string
    type: string
    n_slices: number
    slice_keys: Array<SliceKey>
}


export async function get_info(api_url: string, sliceby_id: string): Promise<SliceByInfo> {
    return await get(`${api_url}/sliceby/${sliceby_id}/info`);
}

export async function get_rows(api_url: string, sliceby_id: string, slice_key: SliceKey, start: number, end: number): Promise<DataPanelRows> {
    return await post(`${api_url}/sliceby/${sliceby_id}/rows`, {slice_key: slice_key, start: start, end: end });
}
