import type { SliceKey } from '$lib/api/sliceby';
import { apply_modifications, get_request, modify, post } from '$lib/utils/requests';
import type { EditTarget } from '$lib/utils/types';
import { get, type Writable } from 'svelte/store';
import { API_URL } from "../constants.js";

export const store_trigger = async (store_id: string, value: any) => {
    const modifications = await modify(`${get(API_URL)}/store/${store_id}/trigger`, { value: value });
    return modifications;
};

export const dispatch = async (endpoint_id: string, kwargs: any, payload: any = {}) => {
    if (endpoint_id === null) {
        return;
    }
    const [result, modifications] = await post(`${get(API_URL)}/endpoint/${endpoint_id}/dispatch`, {
        fn_kwargs: kwargs,
        payload: payload
    });
    apply_modifications(modifications);
    return result;
};

export const get_schema = async (ref_id: string, columns: Array<string> | null = null) => {
    return await post(`${get(API_URL)}/df/${ref_id}/schema`, { columns: columns });
};

export const get_rows = async (
    ref_id: string,
    start?: number,
    end?: number,
    indices?: Array<number>,
    columns?: Array<string>,
    key_column?: string,
    keys?: Array<string | number>
) => {
    const result = await post(`${get(API_URL)}/df/${ref_id}/rows`, {
        start: start,
        end: end,
        indices: indices,
        key_column: key_column,
        keys: keys,
        columns: columns
    });
    return result;
};

export const add = async (ref_id: string, column_name: string) => {
    const modifications = await modify(`${get(API_URL)}/df/${ref_id}/add`, { column: column_name });
    return modifications;
};

export const edit = async (
    ref_id: string,
    value: string | number,
    column: string,
    row_id: string | number,
    id_column: string
) => {
    const modifications = await modify(`${get(API_URL)}/df/${ref_id}/edit`, {
        value: value,
        column: column,
        row_id: row_id,
        id_column: id_column
    });
    return modifications;
};

export const edit_target = async (
    ref_id: string,
    target: EditTarget,
    value: any,
    column: string,
    row_indices: Array<number>,
    row_keys: Array<string>,
    primary_key: string,
    metadata: any
) => {
    const modifications = await modify(`${get(API_URL)}/df/${ref_id}/edit_target`, {
        target: target,
        value: value,
        column: column,
        row_indices: row_indices,
        row_keys: row_keys,
        primary_key: primary_key,
        metadata: metadata
    });
    return modifications;
};

export const match = async (ref_id: string, input: string, query: string, col_out: Writable<string>) => {
    const modifications = await modify(`${get(API_URL)}/ops/${ref_id}/match`, {
        input: input,
        query: query,
        col_out: col_out.store_id
    });
    return modifications;
};

export const get_sliceby_info = async (ref_id: string) => {
    return await get_request(`${get(API_URL)}/sliceby/${ref_id}/info`);
};

export const get_sliceby_rows = async (
    ref_id: string,
    slice_key: SliceKey,
    start?: number,
    end?: number
) => {
    return await post(`${get(API_URL)}/sliceby/${ref_id}/rows`, {
        slice_key: slice_key,
        start: start,
        end: end
    });
};

export const aggregate_sliceby = async (ref_id: string, aggregations: { string: { id: string } }) => {
    const out = Object();
    for (const [name, aggregation] of Object.entries(aggregations)) {
        out[name] = await post(`${get(API_URL)}/sliceby/${ref_id}/aggregate/`, {
            aggregation_id: aggregation.id,
            accepts_df: true
        });
    }
    return out;
};

export const remove_row_by_index = async (ref_id: string, row_index: number) => {
    const modifications = await modify(`${get(API_URL)}/df/${ref_id}/remove_row_by_index`, {
        row_index: row_index
    });
    return modifications;
};