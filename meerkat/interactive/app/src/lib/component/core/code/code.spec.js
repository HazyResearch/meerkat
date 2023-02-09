import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import Code from './Code.svelte';

describe('Code', () => {
    it('should render code', async () => {
        const { container } = render(Code, {
            props: {
                body: 'import meerkat as mk\ndf = mk.DataFrame()\nprint(df)',
            },
        });

        // Expect the code to be rendered
        expect(screen.getByText(/meerkat/)).toBeInTheDocument();
        expect(screen.getByText(/DataFrame()/)).toBeInTheDocument();
        expect(screen.getByText(/print/)).toBeInTheDocument();

        // Expect the snapshot to match
        expect(container).toMatchSnapshot();

    });
});