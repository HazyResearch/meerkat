import { apply_modifications, modify, post } from '$lib/utils/requests';
import toast from 'svelte-french-toast';
import { get } from 'svelte/store';
import { API_URL } from "../constants.js";
import { DataFrameChunk, type DataFrameRef } from './dataframe.js';

export const update_store = async (store_id: string, value: any) => {
    const modifications = await modify(`${get(API_URL)}/store/${store_id}/update`, { value: value });
    return modifications;
};

export const dispatch = async (endpoint_id: string, payload: any = {}) => {
    if (endpoint_id === null) {
        return;
    }
    const promise = post(`${get(API_URL)}/endpoint/${endpoint_id}/dispatch`, payload).catch(
        (error) => {
            console.log(error);
            toast.error(error.message);
            // Pass the error along.
            throw error;
        }
    );
    const { result, modifications, error } = await promise;
    // Code below is executed only if the promise is successful.
    apply_modifications(modifications);
    return result;
};

export interface DataFrameChunkRequest {
    df: DataFrameRef
    columns?: Array<string> | null
    variants?: Array<string> | null
}

export const fetch_schema = async ({
    df,
    columns = null,
    variants = null
}: DataFrameChunkRequest) => {
    return await post(`${get(API_URL)}/df/${df.ref_id}/schema`, {
        columns: columns, variants: variants
    });
}

export interface DataFrameChunkRequest {
    df: DataFrameRef
    columns?: Array<string> | null
    start?: number | null
    end?: number | null
    posidxs?: Array<number> | null
    keyidxs?: Array<string | number> | null
    key_column?: string | null
    variants?: Array<string> | null
}

export const fetch_chunk = async ({
    df,
    columns = null,
    start = null,
    end = null,
    posidxs = null,
    keyidxs = null,
    key_column = null,
    variants = null
}: DataFrameChunkRequest) => {
    const result = await post(`${get(API_URL)}/df/${df.ref_id}/rows`, {
        start: start,
        end: end,
        posidxs: posidxs,
        key_column: key_column,
        keyidxs: keyidxs,
        columns: columns,
        variants: variants
    });

    return new DataFrameChunk(
        result.column_infos,
        result.posidxs,
        result.rows,
        result.full_length,
        result.primary_key
    );
}

export default {
    update_store,
    dispatch,
    fetch_schema,
    fetch_chunk
};
