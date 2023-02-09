import type { ComponentType } from "$lib/utils/types";

export interface Tab {
    label: string;
    id: string;
    component: ComponentType;
}