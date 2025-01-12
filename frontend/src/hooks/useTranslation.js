import { useContext } from 'react';
import { LanguageContext } from '../App';
import translations from '../translations';

const useTranslation = () => {
  const { language } = useContext(LanguageContext);

  const t = (key) => {
    const keys = key.split('.');
    let value = translations[language];
    
    for (const k of keys) {
      if (value && value[k]) {
        value = value[k];
      } else {
        console.warn(`Translation key not found: ${key}`);
        return key;
      }
    }
    
    return value;
  };

  return { t, language };
};

export default useTranslation; 