import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import Text from './Text.svelte';

describe('Text', () => {
    it('should render Text', async () => {
        const { container, component } = render(Text, {
            props: {
                data: 'hello',
            },
        });
        expect(container).toMatchSnapshot();

        // Check that the text is rendered
        const text = screen.getByText('hello');
        expect(text).toBeInTheDocument();
    });
});