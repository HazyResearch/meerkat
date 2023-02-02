import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import Json from './Json.svelte';

describe('Json', () => {
    it('should render json', async () => {
        const { container } = render(Json, {
            props: {
                body: {
                    "hello": "world",
                    "foo": "bar"
                },
            },
        });

        // Find hello using a regex
        const hello = screen.getByText(/hello/, { selector: 'span' });
        expect(hello).toBeTruthy();

        expect(container).toMatchSnapshot();
    });
});