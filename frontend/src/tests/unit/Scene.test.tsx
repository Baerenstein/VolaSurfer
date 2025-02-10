import { render } from '@testing-library/react';
import { Scene } from '../../components/3d/Scene';
import { describe, it, expect } from 'vitest';

describe('Scene Component', () => {
  it('renders without crashing', () => {
    const { container } = render(<Scene />);
    expect(container).toBeInTheDocument();
  });
});