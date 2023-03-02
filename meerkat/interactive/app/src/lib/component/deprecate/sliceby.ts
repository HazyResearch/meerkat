export type SliceKey = string | number

export interface SliceByInfo {
    id: string
    type: string
    n_slices: number
    slice_keys: Array<SliceKey>
}
