export interface FilterCriterion {
    is_enabled: boolean;
    column: string;
    op: string;
    value: any;
    source: string;
    is_fixed: boolean;
}
