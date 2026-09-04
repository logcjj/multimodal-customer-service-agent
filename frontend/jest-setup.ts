import '@testing-library/jest-dom';
import React from 'react';
import { TextDecoder, TextEncoder } from 'node:util';

Object.assign(globalThis, { React, TextDecoder, TextEncoder });
