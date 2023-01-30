export type StatsType = Record<string, number>;

export type InstanceType = {
    input: string;
    tags: Record<string, string>;
    correct: Record<string, boolean>;
};