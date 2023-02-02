import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import Markdown from './Markdown.svelte';

describe('Markdown', () => {
    it('should render markdown', async () => {
        const { container } = render(Markdown, {
            props: {
                body: " # hello This is a **bold** and it has a [link](https://google.com)",
                headerIds: false,
            },
        });

        // Find hello using a regex
        const h1 = screen.getByText(/hello/, { selector: 'h1' });
        expect(h1).toBeTruthy();

        // Expect an h1 tag
        expect(h1.tagName).toBe('H1');

        // Expect a bold tag
        const bold = screen.getByText('bold', { selector: 'strong' });
        expect(bold).toBeTruthy();

        // Expect a link tag
        const link = screen.getByRole('link', { name: 'link' });
        expect(link).toBeTruthy();

        expect(container).toMatchSnapshot();

    });
});