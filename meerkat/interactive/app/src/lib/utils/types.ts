import type { Writable } from "svelte/store";

export interface CellInterface {
    data: any;
    column?: string;
    cell_component?: string;
    cell_props?: object;
}

export type DescriptionType = {
    score: number;
    description: string;
};

export type SliceType = {
    name: string;
    descriptions: Array<{ score: number; description: string; }>;
    stats: StatsType;
    instances: Array<InstanceType>;
    plot: { PLOT_TYPE: string; matrix: undefined | MatrixType; html: string; };
};

export type StatsType = Record<string, number>;

export type InstanceType = {
    input: string;
    tags: Record<string, string>;
    correct: Record<string, boolean>;
};

export interface ComponentType {
    component_id: string;
    path: string;
    name: string;
    props: any;
    slots: any;
    library: string;
}

export interface Endpoint {
    endpoint_id: string;
}

export interface Layout {
    name: string
    props: any;
}

export interface PageType {
    name: string;
    component: ComponentType
}

export interface EditTarget {
    target: Writable<any>
    target_id_column: string
    source_id_column: string
}
