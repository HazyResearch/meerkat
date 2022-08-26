import { post } from '$lib/utils/requests';


export interface CategoryGenerationResponse {
    categories: Array<string>
}

export async function get_categories(
    api_url: string,
    dataset_description: string,
    hint: string,
): Promise<CategoryGenerationResponse> {
    
    console.log(`${api_url}/generate/categories`);
    console.log(`dataset_description: ${dataset_description}`);
    console.log(`hint: ${hint}`);

    return await post(
        `${api_url}/llm/generate/categories`,
        { dataset_description: dataset_description, hint: hint }
    );
}

export async function get_categorization(
    api_url: string,
    description: string,
    existing_categories: Array<string>,
): Promise<CategoryGenerationResponse> {
    console.log(`${api_url}/llm/generate/categorization`);
    console.log(`description: ${description}`);
    console.log(`existing_categories: ${existing_categories}`);

    if (existing_categories.length == 0) {
        return await post(
            `${api_url}/llm/generate/categorization`,
            { description: description }
        );
    }
    return await post(
        `${api_url}/llm/generate/categorization`,
        { description: description, existing_categories: existing_categories }
    );
}
