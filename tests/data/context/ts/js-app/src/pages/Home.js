import React from 'react';
import Header from 'components/Header';
import { fmt } from 'utils';
import { gone } from 'components/Gone';
import logo from 'assets/logo.png';
import './Home.css';

export default function Home() {
  return [React, Header, fmt(1), gone, logo];
}
