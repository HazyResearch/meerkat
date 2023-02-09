import { applyModifications, modify, post } from '$lib/utils/requests';
import toast from 'svelte-french-toast';
import { get } from 'svelte/store';
import { API_URL } from "../constants.js";
import { DataFrameChunk, type DataFrameRef } from './dataframe.js';

export const updateStore = async (storeId: string, value: any) => {
    const modifications = await modify(`${get(API_URL)}/store/${storeId}/update`, { value: value });
    return modifications;
};

export const dispatch = async (endpointId: string, payload: any = {}) => {
    if (endpointId === null) {
        return;
    }
    const promise = post(`${get(API_URL)}/endpoint/${endpointId}/dispatch`, payload).catch(
        (error) => {
            console.log(error);
            toast.error(error.message);
            // Pass the error along.
            throw error;
        }
    );
    const { result, modifications, error } = await promise;
    // Code below is executed only if the promise is successful.
    applyModifications(modifications);
    return result;
};

export interface DataFrameChunkRequest {
    df: DataFrameRef
    columns?: Array<string> | null
    variants?: Array<string> | null
}

export const fetchSchema = async ({
    df,
    columns = null,
    variants = null
}: DataFrameChunkRequest) => {
    return await post(`${get(API_URL)}/df/${df.refId}/schema`, {
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
    keyColumn?: string | null
    variants?: Array<string> | null
}

export const fetchChunk = async ({
    df,
    columns = null,
    start = null,
    end = null,
    posidxs = null,
    keyidxs = null,
    keyColumn = null,
    variants = null
}: DataFrameChunkRequest) => {
    const result = await post(`${get(API_URL)}/df/${df.refId}/rows`, {
        start: start,
        end: end,
        posidxs: posidxs,
        key_column: keyColumn,
        keyidxs: keyidxs,
        columns: columns,
        variants: variants
    });

    return new DataFrameChunk(
        result.columnInfos,
        result.posidxs,
        result.rows,
        result.fullLength,
        result.primaryKey
    );
}

export default {
    updateStore: updateStore,
    dispatch: dispatch,
    fetchSchema: fetchSchema,
    fetchChunk: fetchChunk
};
